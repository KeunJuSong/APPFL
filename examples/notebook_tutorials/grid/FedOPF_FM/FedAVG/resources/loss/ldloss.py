import torch.nn as nn
import torch

class LDLoss:
    """Lagrangian Dual Loss Function"""

    def __init__(self):
        super().__init__()

    def forward(self, acopf_data, input, prediction, LagM_slack_p_gen, LagM_gen, LagM_bus, LagM_line):
        acopf_data._device = input.device # sync the device.
        X_ = torch.zeros((prediction.shape[0], acopf_data.nbus*2)).to(acopf_data.device)
        for idx in range(X_.shape[0]):
            X_[idx,:acopf_data.nbus] = input[acopf_data.nbus*idx:acopf_data.nbus*(idx+1),0] # Pd
            X_[idx,acopf_data.nbus:] = input[acopf_data.nbus*idx:acopf_data.nbus*(idx+1),1] # Qd

        obj_cost = acopf_data.obj_fn(prediction)
        ineq_dist = acopf_data.ineq_dist(X_, prediction)
        eq_resid = acopf_data.eq_resid(X_, prediction)

        ineq_spg = ineq_dist[:,:2]
        ineq_qg = ineq_dist[:,2:2+2*acopf_data.ng]
        ineq_v_m = ineq_dist[:,2+2*acopf_data.ng:2+2*acopf_data.ng+2*acopf_data.nbus]
        ineq_line_l = ineq_dist[:,2+2*acopf_data.ng+2*acopf_data.nbus:]

        ineq_cost_spg = LagM_slack_p_gen*ineq_spg
        ineq_cost_qg = LagM_gen*ineq_qg
        ineq_cost_v_m = LagM_bus*ineq_v_m
        ineq_cost_line_l = LagM_line*ineq_line_l
        ineq_cost = torch.cat([ineq_cost_spg, ineq_cost_qg, ineq_cost_v_m, ineq_cost_line_l], dim=1)

        # return 10*obj_cost + ineq_cost.sum(dim = 1), obj_cost, ineq_dist, eq_resid
        return obj_cost + ineq_cost.sum(dim = 1), obj_cost, ineq_dist, eq_resid
