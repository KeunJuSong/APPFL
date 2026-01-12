import copy
import time
import torch
import wandb
import importlib
import numpy as np
from torch.nn import Module
from omegaconf import DictConfig
from typing import Tuple, Dict, Optional, Any

# from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset
from  torch_geometric.loader import DataLoader

from appfl.privacy import (
    laplace_mechanism_output_perturb,
    gaussian_mechanism_output_perturb,
    make_private_with_opacus,
)
from appfl.algorithm.trainer.base_trainer import BaseTrainer
from appfl.misc.utils import parse_device_str, apply_model_device
from appfl.misc.memory_utils import (
    extract_model_state_optimized,
    safe_inplace_operation,
    optimize_memory_cleanup,
)
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import logging

logging.getLogger().handlers.clear()
logging.getLogger().setLevel(logging.WARNING)

# NOTE: Specific libraries for FedOPF usecase
torch._C._set_linalg_preferred_backend(torch._C._LinalgBackend.Cusolver) # torch._C._LinalgBackend.Magma
torch._C._get_linalg_preferred_backend()


class ACOPFTrainer(BaseTrainer):
    """
    ACOPFTrainer:
        Based on vanilla trainer for FL clients, which trains the model using `torch.optim`
        optimizers for a certain number of local epochs or local steps.
        Users need to specify which training model to use in the configuration,
        as well as the number of local epochs or steps.
    """

    def __init__(
        self,
        model: Optional[Module] = None,
        loss_fn: Optional[Module] = None,
        metric: Optional[Any] = None,
        dataset = None, # NOTE: PowerGrid information for ACOPF 
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,

        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            dataset=dataset,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            **kwargs,
        )
        # Check for optimize_memory in train_configs, default to True
        self.optimize_memory = getattr(train_configs, "optimize_memory", True)

        self.privacy_engine = None
        if not hasattr(self.train_configs, "device"):
            self.train_configs.device = "cpu"
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_configs.get("train_batch_size", 32),
            shuffle=self.train_configs.get("train_data_shuffle", True),
            num_workers=self.train_configs.get("num_workers", 0),
            drop_last=True
        )
        self.val_dataloader = (
            DataLoader(
                self.val_dataset,
                batch_size=self.train_configs.get("val_batch_size", 32),
                shuffle=self.train_configs.get("val_data_shuffle", False),
                num_workers=self.train_configs.get("num_workers", 0),
                drop_last=True
            )
            if self.val_dataset is not None
            else None
        )
        if (
            hasattr(self.train_configs, "enable_wandb")
            and self.train_configs.enable_wandb
        ):
            self.enabled_wandb = True
            self.wandb_logging_id = self.train_configs.wandb_logging_id
        else:
            self.enabled_wandb = False
        self._sanity_check()

        # Extract train device, and configurations for possible DataParallel
        self.device_config, self.device = parse_device_str(self.train_configs.device)

        # Differential privacy through Opacus
        if self.train_configs.get("use_dp", False) and (
            self.train_configs.get("dp_mechanism", "laplace") == "opacus"
        ):
            self.privacy_engine = PrivacyEngine()
        
        # # Put all ACOPF variables in "data" to the cuda.
        # self.dataset._device = self.device # sync the device
        # for attr in dir(self.dataset):
        #     var = getattr(self.dataset, attr)
        #     if not callable(var) and not attr.startswith("__") and torch.is_tensor(var):
        #         try:
        #             # setattr(data, attr, var.to(DEVICE))
        #             setattr(self.dataset, attr, var.to(self.device))
        #         except AttributeError:
        #             pass

    def train(self, **kwargs):
        """
        Train the model for a certain number of local epochs or steps and store the mode state
        (probably with perturbation for differential privacy) in `self.model_state`.
        """
        if "round" in kwargs:
            self.round = kwargs["round"]
        self.val_results = {"round": self.round + 1}

        # Store the previous model state for gradient computation
        send_gradient = self.train_configs.get("send_gradient", False)
        if send_gradient:
            if self.optimize_memory:
                self.model_prev = extract_model_state_optimized(
                    self.model, include_buffers=True, cpu_transfer=False
                )
                # self.model_prev = extract_model_state_optimized(
                #     self.model.layers, include_buffers=True, cpu_transfer=False
                # )
            else:
                self.model_prev = copy.deepcopy(self.model.state_dict())
                # self.model_prev = copy.deepcopy(self.model.layers.state_dict())

        # Configure model for possible DataParallel
        self.model = apply_model_device(self.model, self.device_config, self.device)

        do_validation = (
            self.train_configs.get("do_validation", False)
            and self.val_dataloader is not None
        )
        do_pre_validation = (
            self.train_configs.get("do_pre_validation", False)
            and self.val_dataloader is not None
        )

        # Set up logging title
        title = (
            ["Round", "Time", "Train Loss", "Train Accuracy"]
            if (not do_validation) and (not do_pre_validation)
            else (
                [
                    "Round",
                    "Pre Val?",
                    "Time",
                    "Train Loss",
                    "Train Accuracy",
                    "Val Loss",
                    "Val Accuracy",
                ]
                if do_pre_validation
                else [
                    "Round",
                    "Time",
                    "Train Loss",
                    "Train Accuracy",
                    "Val Loss",
                    "Val Accuracy",
                ]
            )
        )
        if self.train_configs.mode == "epoch":
            title.insert(1, "Epoch")

        if self.round == 0:
            self.logger.log_title(title)
        self.logger.set_title(title)

        if do_pre_validation:
            val_loss, val_accuracy = self._validate()
            self.val_results["pre_val_loss"] = val_loss
            self.val_results["pre_val_accuracy"] = val_accuracy
            content = [self.round, "Y", " ", " ", " ", val_loss, val_accuracy]
            if self.train_configs.mode == "epoch":
                content.insert(1, 0)
            self.logger.log_content(content)
            if self.enabled_wandb:
                wandb.log(
                    {
                        f"{self.wandb_logging_id}/val-loss (before train)": val_loss,
                        f"{self.wandb_logging_id}/val-accuracy (before train)": val_accuracy,
                    }
                )

        # Start training
        optim_module = importlib.import_module("torch.optim")
        assert hasattr(optim_module, self.train_configs.optim), (
            f"Optimizer {self.train_configs.optim} not found in torch.optim"
        )
        optimizer = getattr(optim_module, self.train_configs.optim)(
            self.model.parameters(), **self.train_configs.optim_args
        )

        if self.train_configs.get("use_dp", False) and (
            self.train_configs.get("dp_mechanism", "laplace") == "opacus"
        ):
            dp_cfg = self.train_configs.get("dp_config", {})
            noise_multiplier = dp_cfg.get("noise_multiplier", 1.0)
            max_grad_norm = dp_cfg.get("max_grad_norm", 1.0)

            self.model, optimizer, self.train_dataloader = make_private_with_opacus(
                self.privacy_engine,
                self.model,
                optimizer,
                self.train_dataloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                device=self.train_configs.device,
            )

        if self.train_configs.mode == "epoch":
            train_len_range = (0,len(self.train_dataset)) 
            node_means, node_stds, edge_means, edge_stds = self.dataset.input_standardization(train_len_range)
            n_means = node_means.to(self.device)
            n_stds = node_stds.to(self.device)
            e_means = edge_means.to(self.device)
            e_stds = edge_stds.to(self.device)

            # LDF parameters.
            LagM_sp_gen = torch.ones(1, 2).to(self.device) # shape: (1, num_inequalities)
            LagM_gen = torch.ones(1, 2*self.dataset.ng).to(self.device) # shape: (1, num_inequalities)
            LagM_bus = torch.ones(1, 2*self.dataset.nbus).to(self.device) # shape: (1, num_inequalities)
            LagM_line = torch.ones(1, 2*self.dataset.nl).to(self.device) # shape: (1, num_inequalities)

            warmup_iter = self.train_configs.warmup_iter
            rho_init = self.train_configs.rho_init
            rho = rho_init
            rho_iter = 0
            p_iter_max = self.train_configs.p_iter_max
            p_iter_max_sum = p_iter_max

            d = 0 # 0 for static case otherwise use 5"
            beta = 0 
            if self.round <= warmup_iter:
                # print("warm up!")
                for epoch in range(self.train_configs.num_local_epochs):
                    start_time = time.time()
                    train_loss = np.array([]) # 0
                    for data in self.train_dataloader:
                        loss, pred = self._train_batch(optimizer, data, 
                                                            n_means, n_stds, e_means, e_stds,
                                                            LagM_sp_gen, LagM_gen, LagM_bus, LagM_line)
                        
                        # train_loss += loss
                        train_loss = np.concatenate((train_loss, loss), axis=0)

                    # train_loss /= len(self.train_dataloader)
                    train_loss = np.mean(train_loss)

                    # train_accuracy = float(self.metric(target_true, target_pred))
                    obj, feas_vio_max, feas_vio_mean, feas_sat = self.metric(self.dataset, pred, data.x)
                    train_accuracy = np.mean(feas_sat[1]) # NOTE: just for test!

                    if do_validation:
                        val_loss, val_accuracy = self._validate()
                        if "val_loss" not in self.val_results:
                            self.val_results["val_loss"] = []
                            self.val_results["val_accuracy"] = []
                        self.val_results["val_loss"].append(val_loss)
                        self.val_results["val_accuracy"].append(val_accuracy)
                    per_epoch_time = time.time() - start_time
                    if self.enabled_wandb:
                        wandb.log(
                            {
                                f"{self.wandb_logging_id}/train-loss (during train)": train_loss,
                                f"{self.wandb_logging_id}/train-accuracy (during train)": train_accuracy,
                                f"{self.wandb_logging_id}/val-loss (during train)": val_loss,
                                f"{self.wandb_logging_id}/val-accuracy (during train)": val_accuracy,
                            }
                        )
                    self.logger.log_content(
                        [self.round, epoch, per_epoch_time, train_loss, train_accuracy]
                        if (not do_validation) and (not do_pre_validation)
                        else (
                            [
                                self.round,
                                epoch,
                                per_epoch_time,
                                train_loss,
                                train_accuracy,
                                val_loss,
                                val_accuracy,
                            ]
                            if not do_pre_validation
                            else [
                                self.round,
                                epoch,
                                "N",
                                per_epoch_time,
                                train_loss,
                                train_accuracy,
                                val_loss,
                                val_accuracy,
                            ]
                        )
                    )
            else:
                print("warm up end!")
                for epoch in range(self.train_configs.num_local_epochs):
                    start_time = time.time()
                    train_loss = 0
                    if (self.round)%p_iter_max_sum == 0:
                        ######### Outer Interation: calculate step size of lagrangian multipliers update #########
                        rho = rho_init * (1/(1+beta*(rho_iter + 1))) # 근데 이 부분 중복아닌가?? 있어야 하나????
                        rho_iter += 1
                        with torch.no_grad():
                            self.model.eval()
                            for data in self.train_dataloader:
                                data = data.to(self.device)
                                #solver_opt.zero_grad()
                                Yhat_train = self.model(data, n_means, n_stds, e_means, e_stds) # solve power flow equation using Newton-Rahpson method (NOTE: if "useCompl" is True.)
                                # Yhat_train = solver_net(Xtrain, n_means, n_stds) # solve power flow equation using Newton-Rahpson method (NOTE: if "useCompl" is True.)

                                # LagM += rho*ineq_violation(data, Xtrain.x, Yhat_train) # Update the lagrangian multiplier.
                                LagM_sp_gen += rho*self.ineq_violation(data.x, Yhat_train)[:2] # Update the lagrangian multiplier.
                                LagM_gen += rho*self.ineq_violation(data.x, Yhat_train)[2:2+2*self.dataset.ng] # Update the lagrangian multiplier.
                                LagM_bus += rho*self.ineq_violation(data.x, Yhat_train)[2+2*self.dataset.ng:2+2*self.dataset.ng+2*self.dataset.nbus] # Update the lagrangian multiplier.
                                LagM_line += rho*self.ineq_violation(data.x, Yhat_train)[2+2*self.dataset.ng+2*self.dataset.nbus:] # Update the lagrangian multiplier.

                        ## Does this stablize the learning process much better?
                        LagM_sp_gen = (LagM_sp_gen/len(self.train_dataloader))
                        # LagM_gen = (LagM_gen/len(self.train_dataloader))
                        # LagM_bus = (LagM_bus/len(self.train_dataloader))
                        LagM_line = (LagM_line/len(self.train_dataloader))
                    
                    self.model.train()
                    start_time = time.time()
                    train_loss = np.array([]) # 0
                    for data in self.train_dataloader:
                        loss, pred = self._train_batch(optimizer, data, 
                                                            n_means, n_stds, e_means, e_stds,
                                                            LagM_sp_gen, LagM_gen, LagM_bus, LagM_line)
                        # train_loss += loss
                        train_loss = np.concatenate((train_loss, loss), axis=0)
                    # train_loss /= len(self.train_dataloader)
                    train_loss = np.mean(train_loss)

                    # train_accuracy = float(self.metric(target_true, target_pred))
                    obj, feas_vio_max, feas_vio_mean, feas_sat = self.metric(self.dataset, pred, data.x)
                    train_accuracy = np.mean(feas_sat[1]) # NOTE: just for test!

                    if do_validation:
                        val_loss, val_accuracy = self._validate()
                        if "val_loss" not in self.val_results:
                            self.val_results["val_loss"] = []
                            self.val_results["val_accuracy"] = []
                        self.val_results["val_loss"].append(val_loss)
                        self.val_results["val_accuracy"].append(val_accuracy)
                    per_epoch_time = time.time() - start_time
                    if self.enabled_wandb:
                        wandb.log(
                            {
                                f"{self.wandb_logging_id}/train-loss (during train)": train_loss,
                                f"{self.wandb_logging_id}/train-accuracy (during train)": train_accuracy,
                                f"{self.wandb_logging_id}/val-loss (during train)": val_loss,
                                f"{self.wandb_logging_id}/val-accuracy (during train)": val_accuracy,
                            }
                        )
                    self.logger.log_content(
                        [self.round, epoch, per_epoch_time, train_loss, train_accuracy]
                        if (not do_validation) and (not do_pre_validation)
                        else (
                            [
                                self.round,
                                epoch,
                                per_epoch_time,
                                train_loss,
                                train_accuracy,
                                val_loss,
                                val_accuracy,
                            ]
                            if not do_pre_validation
                            else [
                                self.round,
                                epoch,
                                "N",
                                per_epoch_time,
                                train_loss,
                                train_accuracy,
                                val_loss,
                                val_accuracy,
                            ]
                        )
                    )



        else:
            start_time = time.time()
            train_loss, target_true, target_pred = 0, [], []
            if (
                self.train_configs.get("use_dp", False)
                and self.train_configs.get("dp_mechanism", "laplace") == "opacus"
            ):
                with BatchMemoryManager(
                    data_loader=self.train_dataloader,
                    max_physical_batch_size=self.train_configs.get(
                        "train_batch_size", 32
                    ),
                    optimizer=optimizer,
                ) as memory_safe_data_loader:
                    step_count = 0
                    for data, target in memory_safe_data_loader:
                        loss, pred, label = self._train_batch(optimizer, data, target)
                        train_loss += loss
                        target_true.append(label)
                        target_pred.append(pred)
                        step_count += 1
                        if step_count >= self.train_configs.num_local_steps:
                            break
            else:
                data_iter = iter(self.train_dataloader)
                for _ in range(self.train_configs.num_local_steps):
                    try:
                        data, target = next(data_iter)
                    except:  # noqa E722
                        data_iter = iter(self.train_dataloader)
                        data, target = next(data_iter)
                    loss, pred, label = self._train_batch(optimizer, data, target)
                    train_loss += loss
                    target_true.append(label)
                    target_pred.append(pred)
            train_loss /= len(self.train_dataloader)
            target_true, target_pred = (
                np.concatenate(target_true),
                np.concatenate(target_pred),
            )
            train_accuracy = float(self.metric(target_true, target_pred))
            if do_validation:
                val_loss, val_accuracy = self._validate()
                self.val_results["val_loss"] = val_loss
                self.val_results["val_accuracy"] = val_accuracy
            per_step_time = time.time() - start_time
            if self.enabled_wandb:
                wandb.log(
                    {
                        f"{self.wandb_logging_id}/train-loss (during train)": train_loss,
                        f"{self.wandb_logging_id}/train-accuracy (during train)": train_accuracy,
                        f"{self.wandb_logging_id}/val-loss (during train)": val_loss,
                        f"{self.wandb_logging_id}/val-accuracy (during train)": val_accuracy,
                    }
                )
            self.logger.log_content(
                [self.round, per_step_time, train_loss, train_accuracy]
                if (not do_validation) and (not do_pre_validation)
                else (
                    [
                        self.round,
                        per_step_time,
                        train_loss,
                        train_accuracy,
                        val_loss,
                        val_accuracy,
                    ]
                    if not do_pre_validation
                    else [
                        self.round,
                        "N",
                        per_step_time,
                        train_loss,
                        train_accuracy,
                        val_loss,
                        val_accuracy,
                    ]
                )
            )

        # --- Log DP budget ---
        if (
            self.train_configs.get("use_dp", False)
            and self.train_configs.get("dp_mechanism", "laplace") == "opacus"
        ):
            epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
            self.logger.info(
                f"[DP] Training completed with (ε = {epsilon:.2f}, δ = 1e-5)"
            )

        # If model was wrapped in DataParallel, unload it
        if self.device_config["device_type"] == "gpu-multi":
            self.model = self.model.module.to(self.device)

        # NOTE: For CFL-GP, this need to be modified!
        # self.round += 1

        # Differential privacy
        if self.train_configs.get("use_dp", False) and (
            self.train_configs.get("dp_mechanism", "laplace") == "gaussian"
        ):
            assert hasattr(self.train_configs, "clip_value"), (
                "Using laplace differential privacy, and gradient clipping value must be specified"
            )
            assert hasattr(self.train_configs, "epsilon"), (
                "Using laplace differential privacy, and privacy budget (epsilon) must be specified"
            )
            sensitivity = (
                2.0 * self.train_configs.clip_value * self.train_configs.optim_args.lr
            )
            self.model_state = gaussian_mechanism_output_perturb(
                self.model,
                sensitivity,
                self.train_configs.epsilon,
            )
            # self.model_state = gaussian_mechanism_output_perturb(
            #     self.model.layers,
            #     sensitivity,
            #     self.train_configs.epsilon,
            # )
        elif self.train_configs.get("use_dp", False) and (
            self.train_configs.get("dp_mechanism", "laplace") == "laplace"
        ):
            assert hasattr(self.train_configs, "clip_value"), (
                "Using laplace differential privacy, and gradient clipping value must be specified"
            )
            assert hasattr(self.train_configs, "epsilon"), (
                "Using laplace differential privacy, and privacy budget (epsilon) must be specified"
            )
            sensitivity = (
                2.0 * self.train_configs.clip_value * self.train_configs.optim_args.lr
            )
            self.model_state = laplace_mechanism_output_perturb(
                self.model,
                sensitivity,
                self.train_configs.epsilon,
            )
            # self.model_state = laplace_mechanism_output_perturb(
            #     self.model.layers,
            #     sensitivity,
            #     self.train_configs.epsilon,
            # )
        else:
            if self.optimize_memory:
                self.model_state = extract_model_state_optimized(
                    self.model, include_buffers=True, cpu_transfer=False
                )
                # self.model_state = extract_model_state_optimized(
                #     self.model.layers, include_buffers=True, cpu_transfer=False
                # )
            else:
                self.model_state = copy.deepcopy(self.model.state_dict())
                # self.model_state = copy.deepcopy(self.model.layers.state_dict())

        # Move to CPU for communication
        if "cuda" in self.train_configs.device:
            if self.optimize_memory:
                for k in self.model_state:
                    if self.model_state[k].device.type != "cpu":
                        self.model_state[k] = self.model_state[k].cpu()
                optimize_memory_cleanup(force_gc=True)
            else:
                for k in self.model_state:
                    self.model_state[k] = self.model_state[k].cpu()

        # Compute the gradient if needed
        if send_gradient:
            self._compute_gradient()

    def ineq_violation(self, X, Y):
        X_ = torch.zeros((Y.shape[0], self.dataset.nbus*2)).to(self.device)
        for idx in range(X_.shape[0]):
            X_[idx,:self.dataset.nbus] = X[self.dataset.nbus*idx:self.dataset.nbus*(idx+1),0] # Pd
            X_[idx,self.dataset.nbus:] = X[self.dataset.nbus*idx:self.dataset.nbus*(idx+1),1] # Qd
        ineq_dist = self.dataset.ineq_dist(X_, Y) # (batch, # of constraints)
        ineq_cost = ineq_dist.sum(dim = 0) # (1, # of constraints)
        return ineq_cost


    def get_parameters(self) -> Dict:
        if not hasattr(self, "model_state"):
            if self.optimize_memory:
                self.model_state = extract_model_state_optimized(
                    self.model, include_buffers=True, cpu_transfer=False
                )
                # self.model_state = extract_model_state_optimized(
                #     self.model.layers, include_buffers=True, cpu_transfer=False
                # )
            else:
                self.model_state = copy.deepcopy(self.model.state_dict())
                # self.model_state = copy.deepcopy(self.model.layers.state_dict())
        return (
            (self.model_state, self.val_results)
            if hasattr(self, "val_results")
            else self.model_state
        )

    def _sanity_check(self):
        """
        Check if the configurations are valid.
        """
        assert hasattr(self.train_configs, "mode"), "Training mode must be specified"
        assert self.train_configs.mode in [
            "epoch",
            "step",
        ], "Training mode must be either 'epoch' or 'step'"
        if self.train_configs.mode == "epoch":
            assert hasattr(self.train_configs, "num_local_epochs"), (
                "Number of local epochs must be specified"
            )
        else:
            assert hasattr(self.train_configs, "num_local_steps"), (
                "Number of local steps must be specified"
            )

    def _validate(self) -> Tuple[float, float]:
        """
        Validate the model
        :return: loss, accuracy
        """
        device = self.device
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            target_pred, target_true = [], []
            for data, target in self.val_dataloader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()
                target_true.append(target.detach().cpu().numpy())
                target_pred.append(output.detach().cpu().numpy())
        val_loss /= len(self.val_dataloader)
        val_accuracy = float(
            self.metric(np.concatenate(target_true), np.concatenate(target_pred))
        )
        self.model.train()
        return val_loss, val_accuracy

    def _train_batch(
        self, optimizer: torch.optim.Optimizer, data, n_means, n_stds, e_means, e_stds, LagM_sp_gen, LagM_gen, LagM_bus, LagM_line
    ):
        """
        Train the model for one batch of acopf data
        :param optimizer: torch optimizer
        :param data: input data
        :return: loss, prediction
        """
        device = self.device
        data = data.to(device)
        optimizer.zero_grad()
        output = self.model(data, n_means, n_stds, e_means, e_stds) # solve power flow equation using Newton-Rahpson method (NOTE: if "useCompl" is True.)
        # output = self.model(data)
        loss, obj, ineq_dist, eq_resid = self.loss_fn.forward(self.dataset, data.x, output, LagM_sp_gen, LagM_gen, LagM_bus, LagM_line)
        loss.sum().backward()
        if getattr(self.train_configs, "clip_grad", False):            
            # print("Grad clip is set!!")
            assert hasattr(self.train_configs, "clip_value"), (
                "Gradient clipping value must be specified"
            )
            assert hasattr(self.train_configs, "clip_norm"), (
                "Gradient clipping norm must be specified"
            )
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_configs.clip_value,
                norm_type=self.train_configs.clip_norm,
            )
        optimizer.step()
        # return loss.item(), output.detach().cpu().numpy()
        return loss.detach().cpu().numpy(), output

    def _compute_gradient(self) -> None:
        """
        Compute the gradient of the model and store in `self.model_state`,
        where gradient = prev_model - new_model
        """
        if not hasattr(self, "named_parameters"):
            self.named_parameters = set()
            for name, _ in self.model.named_parameters():
                self.named_parameters.add(name)

        if self.optimize_memory:
            with torch.no_grad():
                for name in self.model_state:
                    if name in self.named_parameters:
                        prev_param = (
                            self.model_prev[name].cpu()
                            if self.model_prev[name].device.type != "cpu"
                            else self.model_prev[name]
                        )
                        self.model_state[name] = safe_inplace_operation(
                            prev_param, "sub", self.model_state[name], alpha=1
                        )
            optimize_memory_cleanup(self.model_prev, force_gc=True)
            del self.model_prev
        else:
            for name in self.model_state:
                if name in self.named_parameters:
                    self.model_state[name] = (
                        self.model_prev[name].cpu() - self.model_state[name]
                    )
