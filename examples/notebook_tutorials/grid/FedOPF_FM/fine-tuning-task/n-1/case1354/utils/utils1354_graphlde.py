import torch
import torch.nn as nn
from torch.autograd import Function
torch.set_default_dtype(torch.float32) # 원래 float64 였음. 그런데 이러면 complex dtype에서 float 128로 너무 커짐

# import torch_sparse
import torch_geometric
from torch_geometric.data import Data 

import numpy as np
import pandas as pd

from scipy.linalg import svd
from scipy.sparse import csc_matrix, coo_matrix

import hashlib
from copy import deepcopy
import scipy.io as spio
import time

from pypower.api import loadcase
from pypower.api import runopf, opf, makeYbus, ext2int, int2ext
from pypower import idx_bus, idx_gen, idx_brch, ppoption, runpf

import random

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ACOPFProblem:
    def __init__(self, grid_filename, contingencies=None, component=None):
        ppc = loadcase(grid_filename)
        ppc = ext2int(ppc)
        self.ppc = ppc
        self.update(contingencies=contingencies, component=component)

    def update(self, contingencies=None, component=None):
        if contingencies == None:
            pass
        else:
            ## change topology of "self.ppc" based on "contingencies" ([{},{},{},...,{}])
            contingency = contingencies[component]
            ppc = deepcopy(self.ppc)
            ## based on the "self.component = {'gen', 'branch'}" and "contingecy (integer that corresponds the index of 'gen' and 'branch')"
            if component == 'gen':
                # ppc[component][contingency,idx_gen.GEN_STATUS] = 0
                bus_id = int(ppc[component][contingency,idx_gen.GEN_BUS])
                ppc[component] = np.delete(ppc[component], contingency, axis=0)
                if ppc['bus'][bus_id,idx_bus.BUS_TYPE] == 2:
                    print("change to load bus!")
                    ppc['bus'][bus_id,idx_bus.BUS_TYPE] = 1
                ppc['gencost'] = np.delete(ppc['gencost'], contingency, axis=0)
            elif component == 'branch':
                # ppc[component][contingency,idx_brch.BR_STATUS] = 0
                ppc[component] = np.delete(ppc[component], contingency, axis=0)
            self.ppc = ext2int(ppc) # reordering indicies & remaining active components for the changed topology 
        
        self.genbase = self.ppc['baseMVA']
        self.baseMVA = self.ppc['baseMVA']

        self.nbus = self.ppc['bus'].shape[0]
        self.ng = self.ppc['gen'].shape[0]
        self.nl = self.ppc['branch'].shape[0]

        self.slack = np.where(self.ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0]
        self.pv = np.where(self.ppc['bus'][:, idx_bus.BUS_TYPE] == 2)[0]
        self.spv = np.concatenate([self.slack, self.pv])
        self.spv.sort()
        self.pq = np.setdiff1d(range(self.nbus), self.spv)
        self.nonslack_idxes = np.sort(np.concatenate([self.pq, self.pv])) # slack bus를 제외한 PV, PQ bus index!

        # indices within gens
        self.slack_ = np.array([np.where(x == self.spv)[0][0] for x in self.slack])
        self.pv_ = np.array([np.where(x == self.spv)[0][0] for x in self.pv])

        self.nslack = len(self.slack)
        self.npv = len(self.pv)

        self.quad_costs = torch.tensor(self.ppc['gencost'][:,4], dtype=torch.get_default_dtype())
        self.lin_costs  = torch.tensor(self.ppc['gencost'][:,5], dtype=torch.get_default_dtype())
        self.const_cost = self.ppc['gencost'][:,6].sum()

        self.pmax = torch.tensor(self.ppc['gen'][:,idx_gen.PMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.pmin = torch.tensor(self.ppc['gen'][:,idx_gen.PMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.qmax = torch.tensor(self.ppc['gen'][:,idx_gen.QMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.qmin = torch.tensor(self.ppc['gen'][:,idx_gen.QMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.vmax = torch.tensor(self.ppc['bus'][:,idx_bus.VMAX], dtype=torch.get_default_dtype())
        self.vmin = torch.tensor(self.ppc['bus'][:,idx_bus.VMIN], dtype=torch.get_default_dtype())
        self.slackva = torch.tensor([np.deg2rad(self.ppc['bus'][self.slack, idx_bus.VA])], 
            dtype=torch.get_default_dtype()).squeeze(-1)

        ## kj--line limit boundary code
        flow_max = (self.ppc['branch'][:, idx_brch.RATE_A] / self.baseMVA)**2
        flow_max[flow_max == 0] = np.inf # np.Inf
        self.line_limit = torch.tensor(flow_max, dtype=torch.get_default_dtype())

        ppc2 = deepcopy(self.ppc)
        Ybus, _, _ = makeYbus(self.baseMVA, ppc2['bus'], ppc2['branch'])
        Ybus = Ybus.todense()
        self.Ybusr = torch.tensor(np.real(Ybus), dtype=torch.get_default_dtype())
        self.Ybusi = torch.tensor(np.imag(Ybus), dtype=torch.get_default_dtype())
    
        self._xdim = self.nbus*2
        # self._ydim = 2*gen.shape[-1] + 2*voltage.shape[-1]
        self._ydim = 2*self.ng + 2*self.nbus
        # self._num = feas_mask.sum()

        self._neq = 2*self.nbus
        self._nineq = 4*self.ng + 2*self.nbus + 2*self.nl
        self._nknowns = self.nslack

        # indices of useful quantities in full solution
        self.pg_start_yidx = 0
        self.qg_start_yidx = self.ng
        self.vm_start_yidx = 2*self.ng
        self.va_start_yidx = 2*self.ng + self.nbus

        ## Define variables and indices for "partial completion" neural network
        # pg (non-slack) and |v|_g (including slack)
        self._partial_vars = np.concatenate([self.pg_start_yidx + self.pv_, self.vm_start_yidx + self.spv, self.va_start_yidx + self.slack])
        # self._other_vars = np.setdiff1d(np.arange(self.ydim), self._partial_vars)
        self._partial_unknown_vars = np.concatenate([self.pg_start_yidx + self.pv_, self.vm_start_yidx + self.spv])

        # initial values for solver
        self.vm_init = np.ones(self.ppc['bus'][:, idx_bus.VM].shape)
        self.va_init = np.zeros(self.ppc['bus'][:, idx_bus.VA].shape)
        
        self.pg_init = self.ppc['gen'][:, idx_gen.PG] / self.genbase
        self.qg_init = self.ppc['gen'][:, idx_gen.QG] / self.genbase

        # voltage angle at slack buses (known)
        self.slack_va = self.va_init[self.slack]

        # indices of useful quantities in partial solution
        self.pg_pv_zidx = np.arange(self.npv)
        self.vm_spv_zidx = np.arange(self.npv, 2*self.npv + self.nslack)

        # useful indices for equality constraints
        self.pflow_start_eqidx = 0
        self.qflow_start_eqidx = self.nbus

        # flag
        #self._eq_converge = False
        self._ineq_converge = None

        ### For Pytorch
        self._device = None

        # trace
        self.Sfmn_tr_list = None
        self.Sfmnbar_tr_list = None
        self.Stmn_tr_list = None
        self.Stmnbar_tr_list = None            

    def check_contingencies(self, component='gen', num_cases=10):
        if num_cases > self.ng and component == 'gen':
            print("The # of contingencies for gen is larger than the # of gen. This normally occurs for small or medium grid.")
            num_cases = self.ng - 1
        if num_cases > self.nl and component == 'branch':
            print("The # of contingencies for branch is larger than the # of branch. This normally occurs for small or medium grid.")
            num_cases = self.nl - 1

        contingencies = {}
        contingencies['contingencies'] = []
        fail_cases = 0
        for case in range(num_cases):
            ppc = deepcopy(self.ppc)
            contingency_dict = {}
            if component == 'gen':
                # randomly choose 1 generator (indicies: 0 ~ (# gen-1) )
                idx = random.randint(0, self.ng-1)
                # set status as 0 (i.e., turn off)
                # ppc[component][idx,idx_gen.GEN_STATUS] = 0
                ppc[component] = np.delete(ppc[component], idx, axis=0)
                ppc['gencost'] = np.delete(ppc['gencost'], idx, axis=0)
            elif component == 'branch':
                # randomly choose 1 branch (indicies: 0 ~ (# branch-1) )
                idx = random.randint(0, self.nl-1)
                # set status as 0 (i.e., cut off)
                # ppc[component][idx,idx_brch.BR_STATUS] = 0
                ppc[component] = np.delete(ppc[component], idx, axis=0)
            else:
                # TODO: Set Error alarm? Exception?
                print("error: N-1 contingency only for gen and branch!")
            
            ppc = ext2int(ppc)
            print(ppc['gen'].shape[0])
            print(ppc['branch'].shape[0])
            # Solver options
            ppopt = ppoption.ppoption(VERBOSE=0)  # OPF_ALG=560 (MIPS PDIPM)
            opf_result = runopf(ppc,ppopt) # runopf(ppc, ppopt)

            if opf_result['success'] == 1:
                contingency_dict[component] = idx
            else:
                # contingency_dict[self.component] = None
                fail_cases += 1

            contingencies['contingencies'] += [contingency_dict]
        contingencies['num_cases'] = num_cases - fail_cases
        return contingencies
    
    def eq_converge(self):
      return self._eq_converge
    
    def ineq_converge(self):
      return self._ineq_converge

    @property
    def partial_vars(self):
        #print("self._partial_vars.shape", self._partial_vars.shape)
        return self._partial_vars

    # @property
    # def other_vars(self):
    #     #print("self._other_vars.shape", self._other_vars.shape)
    #     return self._other_vars

    @property
    def partial_unknown_vars(self):
        #print("self._partial_unknown_vars.shape", self._partial_unknown_vars.shape)
        return self._partial_unknown_vars

    @property
    def xdim(self):
        return self._xdim

    @property
    def ydim(self):
        return self._ydim

    # @property
    # def num(self):
    #     return self._num

    @property
    def neq(self):
        return self._neq

    @property
    def nineq(self):
        return self._nineq

    @property
    def nknowns(self):
        return self._nknowns

    @property
    def device(self):
        return self._device

    def get_yvars(self, Y):
        pg = Y[:, :self.ng]
        qg = Y[:, self.ng:2*self.ng]
        vm = Y[:, -2*self.nbus:-self.nbus]
        va = Y[:, -self.nbus:]
        return pg, qg, vm, va

    def obj_fn(self, Y):
        pg, _, _, _ = self.get_yvars(Y)
        pg_mw = pg * torch.tensor(self.genbase).to(self.device)
        cost = (self.quad_costs * pg_mw**2).sum(axis=1) + \
            (self.lin_costs * pg_mw).sum(axis=1) + \
            self.const_cost
        return cost / (self.genbase.mean() ** 2)
    
    def eq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)

        vr = vm*torch.cos(va)
        vi = vm*torch.sin(va)

        tmp1 = vr@(self.Ybusr.to(dtype=torch.get_default_dtype())) - vi@(self.Ybusi.to(dtype=torch.get_default_dtype()))
        tmp2 = -vr@(self.Ybusi.to(dtype=torch.get_default_dtype())) - vi@(self.Ybusr.to(dtype=torch.get_default_dtype()))

        # real power
        pg_expand = torch.zeros(pg.shape[0], self.nbus, device=self.device)
        pg_expand[:, self.spv] = pg
        real_resid = (pg_expand - X[:, :self.nbus]) - (vr*tmp1 - vi*tmp2)

        # reactive power
        qg_expand = torch.zeros(qg.shape[0], self.nbus, device=self.device)
        qg_expand[:, self.spv] = qg
        react_resid = (qg_expand - X[:, self.nbus:]) - (vr*tmp2 + vi*tmp1)

        ## all residuals
        resids = torch.cat([
            real_resid,
            react_resid
        ], dim=1)
        return resids

    # NOTE: In DC3, the authors do not consider line thermal limit constraints.
    # Thus, the code of line thermal limit const. violation is made by Minsoo Kim.
    def ineq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)

        # Line thermal limit inequality violations!
        _, Yf, Yt = makeYbus(self.baseMVA, self.ppc['bus'], self.ppc['branch'])
        vr = vm*torch.cos(va)
        vi = vm*torch.sin(va)
        vz = torch.complex(vr, vi)

        If = torch.tensor(Yf.todense(), dtype=torch.complex64).to(self.device) @ vz.T
        It = torch.tensor(Yt.todense(), dtype=torch.complex64).to(self.device) @ vz.T
        
        Sf = vz[:,self.ppc['branch'][:,0].astype(int)] * torch.conj(If.T)
        St = vz[:,self.ppc['branch'][:,1].astype(int)] * torch.conj(It.T)
        # apparent power limit, |S|
        Sff = Sf * torch.conj(Sf) 
        Stt = St * torch.conj(St)

        gen_slack_idx = np.where(self.spv == self.slack)[0]
        resids = torch.cat([
            pg[:,gen_slack_idx] - self.pmax[gen_slack_idx], # 1 (for slack)
            self.pmin[gen_slack_idx] - pg[:,gen_slack_idx], # 1 (for slack)

            qg - self.qmax,  # ng
            self.qmin - qg,  # ng

            vm - self.vmax,  # nbus
            self.vmin - vm,  # nbus

            # Line thermal limit inequality violations.
            Sff.real - self.line_limit, # nl
            Stt.real - self.line_limit, # nl

        ], dim=1)

        return resids

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        # result = torch.clamp(resids, 0)
        #self._ineq_converge = torch.max(torch.abs(result))
        return torch.clamp(resids, 0)

    def eq_jac(self, Y):
        _, _, vm, va = self.get_yvars(Y)

        # helper functions
        # mdiag = lambda v1, v2: torch.diag_embed(v1).bmm(torch.diag_embed(v2))
        # Ydiagv = lambda Y, v: Y.unsqueeze(0).expand(v.shape[0], *Y.shape).bmm(torch.diag_embed(v))
        # dtm = lambda v, M: torch.diag_embed(v).bmm(M)

        ## NOTE: DeepLDE+ 내용이 밑에 code 부분을 slightly modify한 것...
        mdiag = lambda v1, v2: torch.diag_embed(v1*v2)
        Ydiagv = lambda Y, v: torch.einsum('ij,bj->bij', Y, v)
        dtm = lambda v, M: torch.einsum('bi,bij->bij', v, M)

        # helper quantities
        #print("va", va)
        cosva = torch.cos(va)
        sinva = torch.sin(va)
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)

        Yr = self.Ybusr.to(dtype=torch.get_default_dtype())
        Yi = self.Ybusi.to(dtype=torch.get_default_dtype())
        YrvrYivi = vr@Yr - vi@Yi
        YivrYrvi = vr@Yi + vi@Yr

        # real power equations
        dreal_dpg = torch.zeros(self.nbus, self.ng, device=self.device) 
        dreal_dpg[self.spv, :] = torch.eye(self.ng, device=self.device)

        dreal_dvm = -mdiag(cosva, YrvrYivi) - dtm(vr, Ydiagv(Yr, cosva)-Ydiagv(Yi, sinva)) \
            -mdiag(sinva, YivrYrvi) - dtm(vi, Ydiagv(Yi, cosva)+Ydiagv(Yr, sinva))
        dreal_dva = -mdiag(-vi, YrvrYivi) - dtm(vr, Ydiagv(Yr, -vi)-Ydiagv(Yi, vr)) \
            -mdiag(vr, YivrYrvi) - dtm(vi, Ydiagv(Yi, -vi)+Ydiagv(Yr, vr))
        
        # reactive power equations
        dreact_dqg = torch.zeros(self.nbus, self.ng, device=self.device)
        dreact_dqg[self.spv, :] = torch.eye(self.ng, device=self.device)
        
        dreact_dvm = mdiag(cosva, YivrYrvi) + dtm(vr, Ydiagv(Yi, cosva)+Ydiagv(Yr, sinva)) \
            -mdiag(sinva, YrvrYivi) - dtm(vi, Ydiagv(Yr, cosva)-Ydiagv(Yi, sinva))
        dreact_dva = mdiag(-vi, YivrYrvi) + dtm(vr, Ydiagv(Yi, -vi)+Ydiagv(Yr, vr)) \
            -mdiag(vr, YrvrYivi) - dtm(vi, Ydiagv(Yr, -vi)-Ydiagv(Yi, vr))

        jac = torch.cat([
            torch.cat([dreal_dpg.unsqueeze(0).expand(vr.shape[0], *dreal_dpg.shape), 
                torch.zeros(vr.shape[0], self.nbus, self.ng, device=self.device), 
                dreal_dvm, dreal_dva], dim=2),
            torch.cat([torch.zeros(vr.shape[0], self.nbus, self.ng, device=self.device), 
                dreact_dqg.unsqueeze(0).expand(vr.shape[0], *dreact_dqg.shape),
                dreact_dvm, dreact_dva], dim=2)],
            dim=1)

        return jac

    # Processes intermediate neural network output
    def process_output(self, X, out):
        # Get ready to reconstruct the variables: Pg, Qg, Vm.
        out2 = nn.Sigmoid()(out[:, :-self.nbus+self.nslack]) # Except the voltage angles area.
        
        # Reconstruct the Pg, Qg, and Vm based on the min-max range (always feasible). 
        pg = out2[:, :self.qg_start_yidx] * self.pmax.to(dtype=torch.get_default_dtype()) + (1-out2[:, :self.qg_start_yidx]) * self.pmin.to(dtype=torch.get_default_dtype())
        qg = out2[:, self.qg_start_yidx:self.vm_start_yidx] * self.qmax.to(dtype=torch.get_default_dtype()) + \
            (1-out2[:, self.qg_start_yidx:self.vm_start_yidx]) * self.qmin.to(dtype=torch.get_default_dtype())
        vm = out2[:, self.vm_start_yidx:] * self.vmax.to(dtype=torch.get_default_dtype()) + (1- out2[:, self.vm_start_yidx:]) * self.vmin.to(dtype=torch.get_default_dtype())

        # Use the prediction values of NN solver for voltage angles except the slack bus (cuz it's already given).
        va = torch.zeros(X.shape[0], self.nbus, device=self.device, dtype=torch.get_default_dtype())
        va[:, self.nonslack_idxes] = out[:, self.va_start_yidx:]
        va[:, self.slack] = torch.tensor(self.slack_va, device=self.device, dtype=torch.get_default_dtype()).unsqueeze(0).expand(X.shape[0], self.nslack)

        return torch.cat([pg, qg, vm, va], dim=1)

    # Solves for the full set of variables
    def complete_partial(self, X, Z):
        #print("Z", Z)
        Y_partial = torch.zeros(Z.shape, device=self.device)

        # Re-scale real parts of Pg
        ### kj--slack bus의 gen을 제외하고 rescale 하는 과정을 제대로 못 수횅하고 있음
        ### self.pmax[1:] <== 57 bus의 경우 slack에 해당하는 gen이 첫번째 index에 있어서 상관이 없지만, 다른 bus case는 첫번째 index에 있지 않음.
        gen_pv_idx = np.where(self.spv != self.slack)[0]
        Y_partial[:, self.pg_pv_zidx] = Z[:, self.pg_pv_zidx] * self.pmax[gen_pv_idx].to(dtype=torch.get_default_dtype()) + \
             (1-Z[:, self.pg_pv_zidx]) * self.pmin[gen_pv_idx].to(dtype=torch.get_default_dtype())

        # Y_partial[:, self.pg_pv_zidx] = Z[:, self.pg_pv_zidx] * self.pmax[1:].to(dtype=torch.get_default_dtype()) + \
        #      (1-Z[:, self.pg_pv_zidx]) * self.pmin[1:].to(dtype=torch.get_default_dtype())
        
        # Re-scale real parts of voltages
        Y_partial[:, self.vm_spv_zidx] = Z[:, self.vm_spv_zidx] * self.vmax[self.spv].to(dtype=torch.get_default_dtype()) + \
            (1-Z[:, self.vm_spv_zidx]) * self.vmin[self.spv].to(dtype=torch.get_default_dtype())
        return PFFunction(self)(X, Y_partial)

    def opt_solve(self, X, solver_type='pypower', tol=1e-4):
        X_np = X.detach().cpu().numpy()

        ppc = self.ppc
        #ppc = int2ext(self.ppc) # 이거 안하면 안풀어짐!

        # Set reduced voltage bounds if applicable
        # ppc['bus'][:,idx_bus.VMIN] = ppc['bus'][:,idx_bus.VMIN] + self.EPS_INTERIOR
        # ppc['bus'][:,idx_bus.VMAX] = ppc['bus'][:,idx_bus.VMAX] - self.EPS_INTERIOR

        # Solver options
        ppopt = ppoption.ppoption(OPF_ALG=560, VERBOSE=0, OPF_VIOLATION=tol)  # MIPS PDIPM

        Y = []
        max_time = 0
        total_time = 0
        total_cost = 0
        for i in range(X_np.shape[0]):
            print(i)
            ppc['bus'][:, idx_bus.PD] = X_np[i, :self.nbus] * self.baseMVA
            ppc['bus'][:, idx_bus.QD] = X_np[i, self.nbus:] * self.baseMVA

            start_time = time.time()
            my_result = runopf(ppc, ppopt)
            total_cost += my_result['f']
            end_time = time.time()
            total_time += (end_time - start_time)
            if end_time - start_time > max_time:
              max_time = end_time - start_time
            pg = my_result['gen'][:, idx_gen.PG] / self.genbase
            qg = my_result['gen'][:, idx_gen.QG] / self.genbase
            vm = my_result['bus'][:, idx_bus.VM]
            va = np.deg2rad(my_result['bus'][:, idx_bus.VA])
            Y.append(np.concatenate([pg, qg, vm, va]))

        return total_cost/(X_np.shape[0]), np.array(Y), total_time, total_time/len(X_np), max_time


# NOTE: kj--Implicit layer theroem for calculating power flow equation!!
def PFFunction(data, tol=1e-2, bsz=5, max_iters=5):
    class PFFunctionFn(Function):
        @staticmethod
        def forward(ctx, X, Z):
            # print(X.shape[0])
            Y = torch.zeros(X.shape[0], data.ydim, device=DEVICE)
            
            # known/estimated values (pg at pv buses, vm at all gens, va at slack bus)
            Y[:, data.pg_start_yidx + data.pv_] = Z[:, data.pg_pv_zidx]    # pg at non-slack gens
            Y[:, data.vm_start_yidx + data.spv] = Z[:, data.vm_spv_zidx]   # vm at gens
            Y[:, data.va_start_yidx + data.slack] = torch.tensor(data.slack_va, device=DEVICE, dtype=torch.get_default_dtype())  # va at slack bus

            # init guesses for remaining values
            Y[:, data.vm_start_yidx + data.pq] = torch.tensor(data.vm_init[data.pq], device=DEVICE, dtype=torch.get_default_dtype())  # vm at load buses
            Y[:, data.va_start_yidx + data.pv] = torch.tensor(data.va_init[data.pv], device=DEVICE, dtype=torch.get_default_dtype())  # va at non-slack gens 
            Y[:, data.va_start_yidx + data.pq] = torch.tensor(data.va_init[data.pq], device=DEVICE, dtype=torch.get_default_dtype())  # va at load buses
            Y[:, data.qg_start_yidx:data.qg_start_yidx+data.ng] = 0    # qg at gens (not used in Newton upd)
            Y[:, data.pg_start_yidx+data.slack_] = 0                   # pg at slack (not used in Newton upd)

            keep_constr = np.concatenate([
                data.pflow_start_eqidx + data.pv,     # real power flow at non-slack gens
                data.pflow_start_eqidx + data.pq,     # real power flow at load buses
                data.qflow_start_eqidx + data.pq])    # reactive power flow at load buses
            newton_guess_inds = np.concatenate([             
                data.vm_start_yidx + data.pq,         # vm at load buses
                data.va_start_yidx + data.pv,         # va at non-slack gens
                data.va_start_yidx + data.pq])        # va at load buses

            converged = torch.zeros(X.shape[0])
            jacs = []
            newton_jacs_inv = []
            
            for b in range(0, X.shape[0], bsz):
                X_b = X[b:b+bsz]
                Y_b = Y[b:b+bsz]

                for i in range(max_iters):
                    # print(i)
                    gy = data.eq_resid(X_b, Y_b)[:, keep_constr]
                    jac_full = data.eq_jac(Y_b) # Output shape: (batch_size, 2*N, 2*N + 2*ng)
                    jac = jac_full[:, keep_constr, :]

                    newton_jac_inv = jac[:, :, newton_guess_inds]
                    delta = torch.linalg.solve(jac[:, :, newton_guess_inds], gy.unsqueeze(-1)).squeeze(-1)

                    Y_b[:, newton_guess_inds] -= delta
                    
                    # if torch.norm(delta, dim=1).abs().max() < tol: # Hard
                    if delta.abs().max(dim = 1).values.mean() < tol: # Soft
                        # print("Converged!")
                        # print(delta.abs().max(dim = 1).values.mean())
                        break
                    # else:
                    #     print("Not Converged..")
                    #     print(delta.abs().max(dim = 1).values.mean())
                
                converged[b:b+bsz] = (delta.abs() < tol).all(dim=1)
                jacs.append(jac_full)
                newton_jacs_inv.append(newton_jac_inv)
                        
            ## Step 2: Solve for remaining variables
            # solve for qg values at all gens (note: requires qg in Y to equal 0 at start of computation)
            Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] = \
                -data.eq_resid(X, Y)[:, data.qflow_start_eqidx + data.spv]
            # solve for pg at slack bus (note: requires slack pg in Y to equal 0 at start of computation)
            Y[:, data.pg_start_yidx + data.slack_] = \
                -data.eq_resid(X, Y)[:, data.pflow_start_eqidx + data.slack]

            ctx.data = data
            ctx.save_for_backward(torch.cat(jacs), torch.cat(newton_jacs_inv),
                torch.tensor(newton_guess_inds, device=DEVICE), 
                torch.tensor(keep_constr, device=DEVICE))

            return Y

        @staticmethod
        def backward(ctx, dl_dy):

            data = ctx.data
            jac, newton_jac_inv, newton_guess_inds, keep_constr = ctx.saved_tensors

            ## Step 2 (calc pg at slack and qg at gens)

            # gradient of all voltages through step 3 outputs
            last_eqs = np.concatenate([data.pflow_start_eqidx + data.slack, data.qflow_start_eqidx + data.spv])
            last_vars = np.concatenate([
                data.pg_start_yidx + data.slack_, np.arange(data.qg_start_yidx, data.qg_start_yidx + data.ng)])
            jac3 = jac[:, last_eqs, :]
            dl_dvmva_3 = -jac3[:, :, data.vm_start_yidx:].transpose(1,2).bmm(
                dl_dy[:, last_vars].unsqueeze(-1)).squeeze(-1)

            # gradient of pd at slack and qd at gens through step 3 outputs
            dl_dpdqd_3 = dl_dy[:, last_vars]

            # insert into correct places in x and y loss vectors
            dl_dy_3 = torch.zeros(dl_dy.shape, device=DEVICE)
            dl_dy_3[:, data.vm_start_yidx:] = dl_dvmva_3

            dl_dx_3 = torch.zeros(dl_dy.shape[0], data.xdim, device=DEVICE)
            dl_dx_3[:, np.concatenate([data.slack, data.nbus + data.spv])] = dl_dpdqd_3

            ## Step 1
            dl_dy_total = dl_dy_3 + dl_dy  # Backward pass vector including result of last step

            # Use precomputed inverse jacobian
            jac2 = jac[:, keep_constr, :]
            
            d_int = torch.linalg.solve(newton_jac_inv.transpose(1,2), dl_dy_total[:,newton_guess_inds].unsqueeze(-1)).squeeze(-1)

            dl_dz_2 = torch.zeros(dl_dy.shape[0], data.npv + data.ng, device=DEVICE)
            dl_dz_2[:, data.pg_pv_zidx] = -d_int[:, :data.npv]  # dl_dpg at pv buses
            dl_dz_2[:, data.vm_spv_zidx] = -jac2[:, :, data.vm_start_yidx + data.spv].transpose(1,2).bmm(
                d_int.unsqueeze(-1)).squeeze(-1)

            dl_dx_2 = torch.zeros(dl_dy.shape[0], data.xdim, device=DEVICE)
            dl_dx_2[:, data.pv] = d_int[:, :data.npv]                       # dl_dpd at pv buses
            dl_dx_2[:, data.pq] = d_int[:, data.npv:data.npv+len(data.pq)]  # dl_dpd at pq buses
            dl_dx_2[:, data.nbus + data.pq] = d_int[:, -len(data.pq):]      # dl_dqd at pq buses


            # Final quantities
            dl_dx_total = dl_dx_3 + dl_dx_2
            dl_dz_total = dl_dz_2 + dl_dy_total[:, np.concatenate([
                data.pg_start_yidx + data.pv_, data.vm_start_yidx + data.spv])]

            return dl_dx_total, dl_dz_total
    return PFFunctionFn.apply