'''
This code is kind of improved version of DeepLDE (i.e, DeepLDE+) for large scale problems.

There are two main changes in this code.
- pytorch einsum function
- modified the terminate code in forward part in PFFunction

Moreover, when considering the large-scale problems, the batch size may need to be minimized.

* This code is also ignore the line thermal limit constraint violations.
'''
from torch.cuda import nvtx
from torch.autograd.function import Function
from torch.profiler import record_function
import sys
import torch
import torch.nn as nn
from torch.autograd import Function
torch.set_default_dtype(torch.float32) # 원래 float64 였음. 그런데 이러면 complex dtype에서 float 128로 너무 커짐

# import torch_sparse
import torch_geometric
from torch_geometric.data import Data 

import numpy as np
import pandas as pd


# import osqp ## NOTE: Operating Spliting Quadractic Programing (maybe next research field..)
# from qpth.qp import QPFunction
#import ipopt
from scipy.linalg import svd

import nvmath.bindings.cudss as cs
from copy import deepcopy
import scipy.io as spio
import time
from pathlib import Path
from pypower.api import loadcase
from pypower.api import runopf, opf, makeYbus, ext2int, int2ext
from pypower import idx_bus, idx_gen, idx_brch, ppoption, runpf
from nvmath import CudaDataType as CD
import cupy as cp
from dataclasses import dataclass
import cupyx.scipy.sparse as cpx
from cupyx.scipy.sparse import csr_matrix as cp_csr, coo_matrix as cp_coo, csc_matrix as cp_csc






### By 수호 : Devie 호출 및 C/C++ CUDA solver 관리 부분 ###
###################################################################

DEVICE = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
# 디바이스별 솔버 보관용이었으나 지금은 하나만 사용 #
_PERSIST_SOLVERS = {}

def get_solver_for_current_device2(reorder_alg=1, fix_pattern=True):
    ### 현재 CUDA 디바이스용 솔버를 전역에서 꺼내오거나, 없으면 생성하는 부분 for cuDSS(선형방정식 풀이)###
    dev = torch.cuda.current_device()
    s = _PERSIST_SOLVERS.get(dev)
    if s is None:
        s = PersistentNewtoncsrcudss(reorder_alg=reorder_alg, fix_pattern=fix_pattern)
        _PERSIST_SOLVERS[dev] = s
    return s

def destroy_all_solvers():
    ### 학습 종료시 모든 솔버를 종료 ###
    for s in list(_PERSIST_SOLVERS.values()):
        try:
            s.destroy()
        except Exception:
            pass
    _PERSIST_SOLVERS.clear()
###################################################################

###################################################################
# ACOPF
###################################################################

class ACOPFProblem:
    """
        minimize_{p_g, q_g, vmag, vang} p_g^T A p_g + b p_g + c
        s.t.                  p_g min   <= p_g  <= p_g max
                              q_g min   <= q_g  <= q_g max
                              vmag min  <= vmag <= vmag max
                              vang_slack = \theta_slack   # voltage angle     
                              (p_g - p_d) + (q_g - q_d)i = diag(vmag e^{i*vang}) conj(Y) (vmag e^{-i*vang})
    """

    def __init__(self, data_filename, grid_filename):
        data = spio.loadmat(data_filename)
        ppc = loadcase(grid_filename)

        if "areas" in list(ppc.keys()):
            del ppc["areas"]
        
        ppc = ext2int(ppc)
        self.nbus = ppc['bus'].shape[0]

        self.ppc = ppc

        self.genbase = ppc['baseMVA']
        self.baseMVA = ppc['baseMVA']

        self.slack = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0]
        self.pv = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 2)[0]
        self.spv = np.concatenate([self.slack, self.pv])
        self.spv.sort()
        self.pq = np.setdiff1d(range(self.nbus), self.spv)
        self.nonslack_idxes = np.sort(np.concatenate([self.pq, self.pv])) # slack bus를 제외한 PV, PQ bus index!

        # indices within gens
        self.slack_ = np.array([np.where(x == self.spv)[0][0] for x in self.slack])
        self.pv_ = np.array([np.where(x == self.spv)[0][0] for x in self.pv])

        self.ng = ppc['gen'].shape[0]
        self.nl = ppc['branch'].shape[0]
        self.nslack = len(self.slack)
        self.npv = len(self.pv)

        self.quad_costs = torch.tensor(ppc['gencost'][:,4], dtype=torch.get_default_dtype())
        self.lin_costs  = torch.tensor(ppc['gencost'][:,5], dtype=torch.get_default_dtype())
        self.const_cost = ppc['gencost'][:,6].sum()

        self.pmax = torch.tensor(ppc['gen'][:,idx_gen.PMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.pmin = torch.tensor(ppc['gen'][:,idx_gen.PMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.qmax = torch.tensor(ppc['gen'][:,idx_gen.QMAX] / self.genbase, dtype=torch.get_default_dtype())
        self.qmin = torch.tensor(ppc['gen'][:,idx_gen.QMIN] / self.genbase, dtype=torch.get_default_dtype())
        self.vmax = torch.tensor(ppc['bus'][:,idx_bus.VMAX], dtype=torch.get_default_dtype())
        self.vmin = torch.tensor(ppc['bus'][:,idx_bus.VMIN], dtype=torch.get_default_dtype())
        self.slackva = torch.tensor([np.deg2rad(ppc['bus'][self.slack, idx_bus.VA])], 
            dtype=torch.get_default_dtype()).squeeze(-1)

        ## kj--line limit boundary code
        flow_max = (ppc['branch'][:, idx_brch.RATE_A] / self.baseMVA)**2
        flow_max[flow_max == 0] = np.inf
        self.line_limit = torch.tensor(flow_max, dtype=torch.get_default_dtype())

        ppc2 = deepcopy(ppc)
        Ybus_csr, _, _ = makeYbus(self.baseMVA, ppc2['bus'], ppc2['branch'])
        Ybus = Ybus_csr.todense()
        self.Ybusr = torch.tensor(np.real(Ybus), dtype=torch.get_default_dtype())
        self.Ybusi = torch.tensor(np.imag(Ybus), dtype=torch.get_default_dtype())
        self.Yr_cp = cp.from_dlpack(self.Ybusr.to(DEVICE))
        self.Yi_cp = cp.from_dlpack(self.Ybusi.to(DEVICE))
        self.Yr_dense = cp.from_dlpack(self.Ybusr.to(DEVICE))
        self.Yi_dense = cp.from_dlpack(self.Ybusi.to(DEVICE))


        self.Yr_csr_cp = cp_csr(cp.from_dlpack(self.Ybusr.to(DEVICE)))
        self.Yi_csr_cp = cp_csr(cp.from_dlpack(self.Ybusi.to(DEVICE)))
        self.YrT_csc_cp = cp_csc(self.Yr_csr_cp.T)  # (nbus, nbus)
        self.YiT_csc_cp = cp_csc(self.Yi_csr_cp.T)

        self.Ybus_coo = Ybus_csr.tocoo()
        diag_idx = cp.arange(self.Yr_cp.shape[0])
        self.Gii_cp = self.Yr_cp[diag_idx, diag_idx]  # (n,)
        self.Bii_cp = self.Yi_cp[diag_idx, diag_idx]  # (n,)

        ###################################################
        #### dataset의 크기에 따라서 맞춰 수정해야함. 7336 제외 -> 바로 아래 / 7336은 해당하는 부분으로  ####
        #### By 수호 ####
        demand = data['Dem'].T / self.baseMVA
        gen =  data['Gen'].T / self.genbase
        voltage = data['Vol'].T


        ### for 7336 (원래 이름은 7678) dataset ###
        # demand = data['Dem_merge'].T / self.baseMVA
        # gen = data['Gen_merge'].T / self.genbase
        # voltage = data['Vol_merge'].T

        ###################################################

        # Check the NaN data and remove it
        feas_mask = ~np.isnan(demand).any(axis=1)
        demand = demand[feas_mask]
        gen = gen[feas_mask]
        voltage = voltage[feas_mask]

        ## Graph data representation
        node_feat_x = np.zeros((demand.shape[0], self.nbus, 2))
        node_mask_x = np.zeros((demand.shape[0], self.nbus))
        for s in range(node_feat_x.shape[0]):
            node_feat_x[s,:,0] = np.real(demand[s,:])
            node_feat_x[s,:,1] = np.imag(demand[s,:])
            node_mask_x[s,self.spv] = 1
            # node_mask_x[s,self.pv] = 1

        edge_feat_x = np.zeros((demand.shape[0], self.nl, 6))
        for s in range(edge_feat_x.shape[0]):
            edge_feat_x[s,:,0] = ppc['branch'][:, idx_brch.F_BUS]
            edge_feat_x[s,:,1] = ppc['branch'][:, idx_brch.T_BUS]
            edge_feat_x[s,:,2] = ppc['branch'][:, idx_brch.BR_R]
            edge_feat_x[s,:,3] = ppc['branch'][:, idx_brch.BR_X]
            edge_feat_x[s,:,4] = ppc['branch'][:, idx_brch.BR_B]
            
            br_ratea_feat = (ppc['branch'][:, idx_brch.RATE_A] / self.baseMVA)
            # br_ratea_feat[br_ratea_feat == 0] = np.Inf
            edge_feat_x[s,:,5] = br_ratea_feat
        
        # NOTE: Please note that this variable is not the exact ouput of DeepLDE's neural networks (exact output is [Pg, Vm]). This is for the some of decision variables in ACOPF.
        # e.g., for 57 bus case, then the Y dim is 7+7+57+57=128.
        node_feat_y = np.zeros((demand.shape[0], self.nbus, 4))
        for s in range(node_feat_y.shape[0]):
            node_feat_y[s,self.spv,0] = np.real(gen[s,:]) # P
            node_feat_y[s,self.spv,1] = np.imag(gen[s,:]) # Q
            node_feat_y[s,:,2] = np.abs(voltage[s,:]) # V
            node_feat_y[s,:,3] = np.angle(voltage[s,:]) # Theta
        
        feas_mask =  ~np.isnan(node_feat_y).any(axis=(1,2)) # Only get the feasible solutions..        

        node_feat_x = torch.tensor(node_feat_x[feas_mask,:,:], dtype=torch.get_default_dtype())
        node_feat_y = torch.tensor(node_feat_y[feas_mask,:,:], dtype=torch.get_default_dtype())
        node_mask_x = torch.tensor(node_mask_x[feas_mask,:], dtype=torch.get_default_dtype())
        edge_feat_x = torch.tensor(edge_feat_x[feas_mask,:,:], dtype=torch.get_default_dtype())
        pyg_data_list = []
        for i in range(demand.shape[0]):
            pyg_data_list += [Data(
                x=node_feat_x[i,:,:],
                y=node_feat_y[i,:,:],
                node_mask=node_mask_x[i,:].to(torch.bool),
                edge_index=edge_feat_x[i, :, 0:2].T.to(torch.long),
                edge_attr=edge_feat_x[i, :, 2:],) # R,X,B,RateA
                ]
        
        # Graph representation data (including x and y)
        self._G_data = pyg_data_list

        self._xdim = self.nbus*2
        self._ydim = 2*gen.shape[-1] + 2*voltage.shape[-1]
        self._num = feas_mask.sum()

        #self._Ypseudo = torch.tensor(Ypseudo, dtype=torch.get_default_dtype())

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
        self._other_vars = np.setdiff1d(np.arange(self.ydim), self._partial_vars)
        self._partial_unknown_vars = np.concatenate([self.pg_start_yidx + self.pv_, self.vm_start_yidx + self.spv])

        # initial values for solver
        # self.vm_init = ppc['bus'][:, idx_bus.VM]
        # self.va_init = np.deg2rad(ppc['bus'][:, idx_bus.VA])
        self.vm_init = np.ones(ppc['bus'][:, idx_bus.VM].shape)
        self.va_init = np.zeros(ppc['bus'][:, idx_bus.VA].shape)
        
        self.pg_init = ppc['gen'][:, idx_gen.PG] / self.genbase
        self.qg_init = ppc['gen'][:, idx_gen.QG] / self.genbase

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


        #### By 수호 ####
        ############### 이 아래는 학습에 사용되는 slicing array들을 선언하는 부분 ###############
        ############## numpy로 유지해서 특히 backward에서는 cupy를 사용해 indexing 할 수 있도록. ################
        ############## jacobian matrix를 slice하는 array는 cupy 로 만들어야 함. ################
        ###################################################################
        # trace
        self.B = None
        self._precompute_jac_singleblock_pattern()
        self._big_jac_ready_B = set() # prepare_big_jac_buffers()가 끝난 B들의 기록
        self.pattern_J_red   = Coo2CsrPatternGPU()  # forward: reduced J
        self.pattern_J_red_T = Coo2CsrPatternGPU()  # backward: J_red^T
        self.pattern_J3_T    = Coo2CsrPatternGPU()  # backward: J_3^T
        self.pattern_J2_T    = Coo2CsrPatternGPU()  # backward: J_2^T
        ###################################################################

    def __str__(self):
        return 'ACOPF-{}-{}-{}-{}-{}-{}'.format(
            self.nbus,
            # self.EPS_INTERIOR, self.CorrCoeff, self.MaxChangeLoad,
            self.valid_frac, self.test_frac)

    # def Smax2(self):
    #     return torch.tensor(np.square(self.ppc['branch'][:,idx_brch.RATE_A] / self.baseMVA), dtype=torch.get_default_dtype(), device=self.device)

    def eq_converge(self):
      return self._eq_converge
    
    def ineq_converge(self):
      return self._ineq_converge

    def input_standardization(self, data_len, train=True):
        if train:
            train_data = self.train_dataset[data_len[0]:data_len[1]]

            node_dataset = torch.zeros((self.nbus*len(train_data),2))
            edge_dataset = torch.zeros((self.nl*len(train_data),4))
            for (i,data) in enumerate(train_data):
                node_dataset[self.nbus*i:self.nbus*(i+1),:] = data.x
                edge_dataset[self.nl*i:self.nl*(i+1),:] = data.edge_attr
            
            node_means = node_dataset.mean(axis=0, keepdims = True)
            node_stds = node_dataset.std(axis=0, keepdims = True)
            edge_means = edge_dataset.mean(axis=0, keepdims = True)
            edge_stds = edge_dataset.std(axis=0, keepdims = True)
        else:
            test_data = self.test_dataset[data_len:]

            node_dataset = torch.zeros((self.nbus*len(test_data),2))
            edge_dataset = torch.zeros((self.nl*len(test_data),4))
            for (i,data) in enumerate(test_data):
                node_dataset[self.nbus*i:self.nbus*(i+1),:] = data.x
                edge_dataset[self.nl*i:self.nl*(i+1),:] = data.edge_attr
            
            node_means = node_dataset.mean(axis=0, keepdims = True)
            node_stds = node_dataset.std(axis=0, keepdims = True)
            edge_means = edge_dataset.mean(axis=0, keepdims = True)
            edge_stds = edge_dataset.std(axis=0, keepdims = True)

        # means = self._X.mean(axis = 0, keepdims = True) # (2*nbus)
        # stds = self._X.std(axis = 0, keepdims = True)
        return node_means.to(dtype=torch.get_default_dtype()), node_stds.to(dtype=torch.get_default_dtype()), edge_means.to(dtype=torch.get_default_dtype()), edge_stds.to(dtype=torch.get_default_dtype())

    @property
    def graph_dataset(self):
        return self._G_data
    
    @property
    def partial_vars(self):
        #print("self._partial_vars.shape", self._partial_vars.shape)
        return self._partial_vars

    @property
    def other_vars(self):
        #print("self._other_vars.shape", self._other_vars.shape)
        return self._other_vars

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

    @property
    def num(self):
        return self._num

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
    def train_frac_semi(self):
        return self._valid_frac

    @property
    def valid_frac(self):
        return self._valid_frac

    @property
    def test_frac(self):
        return self._test_frac

    @property
    def train_frac(self):
        return 1 - self.valid_frac - self.test_frac

    # NOTE: First, just generate tons of dataset and adaptively use among them. e.g., select 12,200 samples among 1e5 samples. --kj 
    # In empiricial way, at least 1e5 samples are proper to make the total # of samples as 12,200 from DeepLDE paper.
    # But just for the test, let's consider the size of test in the kind of remaining way by removing the infeasible results.  
    @property
    def train_dataset(self):
        return self._G_data[:800]

    @property
    def valid_dataset(self):
        return self._G_data[800:1000]

    @property
    def test_dataset(self):
        return self._G_data[800:]

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
        cost = (self.quad_costs.to(self.device) * pg_mw**2).sum(axis=1) + \
            (self.lin_costs.to(self.device) * pg_mw).sum(axis=1) + \
            self.const_cost
        return cost / (self.genbase.mean() ** 2)

    # def cal_Smn(self, Y_mn, Ybar_mn, vm, va):
    #     cosva = torch.cos(va)
    #     sinva = torch.sin(va)
        
    #     vr = vm*cosva # (batch, nbus)
    #     vi = vm*sinva # (batch, nbus)

    #     v = torch.cat([vr, vi], dim = -1) # (batch, 2*nbus)
    #     #print("v.shape", v.shape)
    #     #print("v.shape", v.shape)
        
    #     Smn_tr = v.unsqueeze(-1).transpose(1,2).bmm(Y_mn.matmul(v.unsqueeze(-1))).squeeze(-1) # (batch, 1)

    #     Smnbar_tr = v.unsqueeze(-1).transpose(1,2).bmm(Ybar_mn.matmul(v.unsqueeze(-1))).squeeze(-1) # (batch, 1)
        
    #     return Smn_tr, Smnbar_tr # (batch, 1)    
    
    def eq_resid(self, X, Y):
        pg, qg, vm, va = self.get_yvars(Y)

        vr = vm*torch.cos(va)
        vi = vm*torch.sin(va)

        tmp1 = vr@(self.Ybusr.to(device=self.device, dtype=torch.get_default_dtype())) - vi@(self.Ybusi.to(device=self.device, dtype=torch.get_default_dtype()))
        tmp2 = -vr@(self.Ybusi.to(device=self.device, dtype=torch.get_default_dtype())) - vi@(self.Ybusr.to(device=self.device, dtype=torch.get_default_dtype()))

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

        '''
        if torch.max(torch.abs(resids)) > eps_converge:
          self._eq_converge = False
        else:
          self._eq_converge = True
        '''
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
            pg[:,gen_slack_idx] - self.pmax[gen_slack_idx].to(self.device), # 1 (for slack)
            self.pmin[gen_slack_idx].to(self.device) - pg[:,gen_slack_idx], # 1 (for slack)

            qg - self.qmax.to(self.device),  # ng
            self.qmin.to(self.device) - qg,  # ng

            vm - self.vmax.to(self.device),  # nbus
            self.vmin.to(self.device) - vm,  # nbus

            # Line thermal limit inequality violations.
            Sff.real - self.line_limit.to(self.device), # nl
            Stt.real - self.line_limit.to(self.device), # nl

        ], dim=1)

        return resids

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        # result = torch.clamp(resids, 0)
        #self._ineq_converge = torch.max(torch.abs(result))
        return torch.clamp(resids, 0)


    '''
    def ineq_resid_back(self, X, Y): # using only for backpropagation
        pg, qg, vm, va = self.get_yvars(Y)
        sf, st = self.cal_S(self.ppc['branch'][:,[0,1]] - 1, self.nl, vm, va)
        resids = torch.cat([
            pg - self.pmax,
            self.pmin - pg,
            qg - self.qmax,
            self.qmin - qg,
            vm - self.vmax,
            self.vmin - vm,
            sf - smax2,
            st - smax2
        ], dim=1)
        return resids
    '''
    #def ineq_dist_back(self, X, Y): # using only for backpropagation
    #    result = self.ineq_dist(X, Y, self.Sfmn_tr_list, self.Sfmnbar_tr_list, self.Stmn_tr_list, self.Stmnbar_tr_list)
    #    return result

    def ineq_dist_back(self, X, Y): # using only for backpropagation

        result = self.ineq_dist(X, Y)
        return result

    def eq_jac(self, Y):
        _, _, vm, va = self.get_yvars(Y)
        
        #def get_yvars(self, Y):
        #    pg = Y[:, :self.ng]
        #    qg = Y[:, self.ng:2*self.ng]
        #    vm = Y[:, -2*self.nbus:-self.nbus]
        #    va = Y[:, -self.nbus:]
        #    return pg, qg, vm, va
        ## input으로 들어오는 Y에 대해서 지금 네 부분으로 나눔. 이때, 맨 앞 pg, qg는 각각 generator bus의 Pg, Qg에 해당. 

        # helper functions
        # mdiag = lambda v1, v2: torch.diag_embed(v1).bmm(torch.diag_embed(v2))
        # Ydiagv = lambda Y, v: Y.unsqueeze(0).expand(v.shape[0], *Y.shape).bmm(torch.diag_embed(v))
        # dtm = lambda v, M: torch.diag_embed(v).bmm(M)

        ## NOTE: DeepLDE+ 내용이 밑에 code 부분을 slightly modify한 것...
        mdiag = lambda v1, v2: torch.diag_embed(v1*v2)  # diag(v1*v2) (B, nbus, nbus)
        Ydiagv = lambda Y, v: torch.einsum('ij,bj->bij', Y, v)  # Y @ diag(v) (B, nbus, nbus)
        dtm = lambda v, M: torch.einsum('bi,bij->bij', v, M)  # diag(v) @ M (B, nbus, nbus)
        ### 지금 vm 과 va 는 각각 nbus, nbus shape을 가짐. 
        ### 그리고 이건 모두 batch를 고려한 size가 도출됨. 
        
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
        dreal_dpg = torch.zeros(self.nbus, self.ng, device=self.device) ## pg가 지금 nbus , ng shape. 즉, 
        dreal_dpg[self.spv, :] = torch.eye(self.ng, device=self.device)
        ## 이 위는 주입 변수에 대한 미분. 즉, nbus, ng였다가 spv에 해당하는. 즉, gen, gen의 크기를 가지게 됨. 
        dreal_dvm = -mdiag(cosva, YrvrYivi) - dtm(vr, Ydiagv(Yr, cosva)-Ydiagv(Yi, sinva)) \
            -mdiag(sinva, YivrYrvi) - dtm(vi, Ydiagv(Yi, cosva)+Ydiagv(Yr, sinva))
        dreal_dva = -mdiag(-vi, YrvrYivi) - dtm(vr, Ydiagv(Yr, -vi)-Ydiagv(Yi, vr)) \
            -mdiag(vr, YivrYrvi) - dtm(vi, Ydiagv(Yi, -vi)+Ydiagv(Yr, vr))



        # reactive power equations
        dreact_dqg = torch.zeros(self.nbus, self.ng, device=self.device)
        dreact_dqg[self.spv, :] = torch.eye(self.ng, device=self.device)
        ## 이 위는 주입 qg에 대한 미분. 즉, nbus, ng였다가 spv에 해당하는. 즉, gen, gen의 크기를 가지게 됨.
        dreact_dvm = mdiag(cosva, YivrYrvi) + dtm(vr, Ydiagv(Yi, cosva)+Ydiagv(Yr, sinva)) \
            -mdiag(sinva, YrvrYivi) - dtm(vi, Ydiagv(Yr, cosva)-Ydiagv(Yi, sinva))
        dreact_dva = mdiag(-vi, YivrYrvi) + dtm(vr, Ydiagv(Yi, -vi)+Ydiagv(Yr, vr)) \
            -mdiag(vr, YrvrYivi) - dtm(vi, Ydiagv(Yr, -vi)-Ydiagv(Yi, vr))

        ### 앞에는 dreal_dpg -> 1,793,89 / torch.zero -> 1,793,89 / torch.Size([1, 793, 793]) / torch.Size([1, 793, 793])
        jac = torch.cat([
            torch.cat([dreal_dpg.unsqueeze(0).expand(vr.shape[0], *dreal_dpg.shape), 
                torch.zeros(vr.shape[0], self.nbus, self.ng, device=self.device), 
                dreal_dvm, dreal_dva], dim=2),
            torch.cat([torch.zeros(vr.shape[0], self.nbus, self.ng, device=self.device), 
                dreact_dqg.unsqueeze(0).expand(vr.shape[0], *dreact_dqg.shape),
                dreact_dvm, dreact_dva], dim=2)],
            dim=1)

        
        ## 최종적으로 만들어지는 jac의 shape = torch.Size([50, 1586, 1764])
        ## torch.Size([50, 1586, 1764])
        ## 793 89
        ## torch.Size([793, 89]) torch.Size([50, 793, 793]) torch.Size([50, 793, 793])
        ## 즉, 이때 
        return jac
    



    #### By 수호 : 가속기에 들어갈 여러 함수를 정의 ####
    ##################################################################

    def _precompute_jac_singleblock_pattern(self, epsY: float = 0.0, force_diag: bool = True):
        """
        한 샘플(2*nbus x 2*ng+2*nbus) 자코비안의 COO 패턴/메타 생성.
        - Y의 실/허수 COO 합집합을 전압 블록(P_VM,P_VA,Q_VM,Q_VA) 패턴으로 사용
        - (row, col, blk_type) 기준 canonical 정렬 & 중복 제거
        - (blk_type, bus_i, bus_j) -> k 룩업 테이블 생성
        """

        nbus = int(self.nbus)
        ng   = int(self.ng)
        spv  = torch.as_tensor(self.spv, dtype=torch.long)

        # === 1) Y 패턴(합집합) ===
        Yr_coo = self.Yr_csr_cp.tocoo()
        Yi_coo = self.Yi_csr_cp.tocoo()

        rG, cG, vG = Yr_coo.row, Yr_coo.col, Yr_coo.data
        rB, cB, vB = Yi_coo.row, Yi_coo.col, Yi_coo.data

        if epsY > 0.0:
            mG = cp.abs(vG) > epsY
            mB = cp.abs(vB) > epsY
            rG, cG, vG = rG[mG], cG[mG], vG[mG]
            rB, cB, vB = rB[mB], cB[mB], vB[mB]

        keysG = (rG.astype(cp.int64) * nbus + cG.astype(cp.int64)).get()
        keysB = (rB.astype(cp.int64) * nbus + cB.astype(cp.int64)).get()

        if force_diag:
            diag_keys = (np.arange(nbus, dtype=np.int64) * nbus + np.arange(nbus, dtype=np.int64))
            keys = np.union1d(np.union1d(keysG, keysB), diag_keys)
        else:
            keys = np.union1d(keysG, keysB)

        # 딕셔너리 (numpy로)
        mapG = {int(k): float(val) for k, val in zip(keysG, vG.get())}
        mapB = {int(k): float(val) for k, val in zip(keysB, vB.get())}

        # === 2) 오프셋 (블록 내부) ===
        vm_off = 2*ng
        va_off = 2*ng + nbus

        # === 3) 패턴 메타 조립 ===
        rows, cols, blk, bi, bj, gval, bval = [], [], [], [], [], [], []

        # P wrt PG (diag at spv)
        for k in range(ng):
            i = int(spv[k])
            rows.append(i);        cols.append(k);          blk.append(4)  # P_PG
            bi.append(i);          bj.append(i);           gval.append(0.0); bval.append(0.0)

        # Q wrt QG (diag at spv)
        for k in range(ng):
            i = int(spv[k])
            rows.append(nbus+i);   cols.append(ng+k);      blk.append(5)  # Q_QG
            bi.append(i);          bj.append(i);           gval.append(0.0); bval.append(0.0)

        # 전압 블록 (합집합 키)
        # keys 를 (i,j)로 정렬해 안정화
        ij = np.array([(int(k // nbus), int(k % nbus)) for k in keys], dtype=np.int64)
        order_ij = np.lexsort((ij[:,1], ij[:,0]))
        ij = ij[order_ij]

        for i, j in ij:
            K = i*nbus + j
            Gij = mapG.get(int(K), 0.0)
            Bij = mapB.get(int(K), 0.0)

            # P_VM
            rows.append(i);        cols.append(vm_off + j); blk.append(0)
            bi.append(i);          bj.append(j);            gval.append(Gij); bval.append(Bij)

            # P_VA
            rows.append(i);        cols.append(va_off + j); blk.append(1)
            bi.append(i);          bj.append(j);            gval.append(Gij); bval.append(Bij)

            # Q_VM
            rows.append(nbus+i);   cols.append(vm_off + j); blk.append(2)
            bi.append(i);          bj.append(j);            gval.append(Gij); bval.append(Bij)

            # Q_VA
            rows.append(nbus+i);   cols.append(va_off + j); blk.append(3)
            bi.append(i);          bj.append(j);            gval.append(Gij); bval.append(Bij)

        # === 4) 텐서화 ===
        rows_t = torch.tensor(rows, dtype=torch.long)
        cols_t = torch.tensor(cols, dtype=torch.long)
        blk_t  = torch.tensor(blk,  dtype=torch.int8)
        bi_t   = torch.tensor(bi,   dtype=torch.long)
        bj_t   = torch.tensor(bj,   dtype=torch.long)
        gval_t = torch.tensor(gval, dtype=torch.float32)
        bval_t = torch.tensor(bval, dtype=torch.float32)

        # === 5) canonical 정렬 + 중복제거 ===
        # numpy lexsort로 안정 정렬
        key_row = rows_t.cpu().numpy()
        key_col = cols_t.cpu().numpy()
        key_blk = blk_t.cpu().numpy().astype(np.int16)
        ord3 = np.lexsort((key_blk, key_col, key_row))

        rows_t = rows_t[torch.tensor(ord3, dtype=torch.long)]
        cols_t = cols_t[torch.tensor(ord3, dtype=torch.long)]
        blk_t  = blk_t [torch.tensor(ord3, dtype=torch.long)]
        bi_t   = bi_t  [torch.tensor(ord3, dtype=torch.long)]
        bj_t   = bj_t  [torch.tensor(ord3, dtype=torch.long)]
        gval_t = gval_t[torch.tensor(ord3, dtype=torch.long)]
        bval_t = bval_t[torch.tensor(ord3, dtype=torch.long)]

        # 연속 중복 제거
        trip = np.stack([rows_t.cpu().numpy(),
                        cols_t.cpu().numpy(),
                        blk_t.cpu().numpy().astype(np.int64)], axis=1)
        keep = np.ones(trip.shape[0], dtype=bool)
        if trip.shape[0] > 1:
            eq_prev = np.all(trip[1:] == trip[:-1], axis=1)
            keep[1:] = ~eq_prev

        keep_t = torch.from_numpy(keep)
        rows_t = rows_t[keep_t]
        cols_t = cols_t[keep_t]
        blk_t  = blk_t [keep_t]
        bi_t   = bi_t  [keep_t]
        bj_t   = bj_t  [keep_t]
        gval_t = gval_t[keep_t]
        bval_t = bval_t[keep_t]

        # === 6) 상태 저장 ===
            # === 6) 상태 저장 ===
        self.jac_rows_single = rows_t
        self.jac_cols_single = cols_t
        self.jac_blk_type    = blk_t
        self.jac_bus_i       = bi_t
        self.jac_bus_j       = bj_t
        self.jac_G_ij        = gval_t
        self.jac_B_ij        = bval_t

        self.jac_m_rows      = 2*nbus
        self.jac_n_cols      = 2*ng + 2*nbus
        self.jac_nnz_single  = self.jac_rows_single.numel()

        # === 7) PG/QG 위치 (blk_type 기반) ===
        self.single_pos_pg_torch = (self.jac_blk_type == 4).nonzero(as_tuple=False).squeeze(1)
        self.single_pos_qg_torch = (self.jac_blk_type == 5).nonzero(as_tuple=False).squeeze(1)

        # === 7-1) VM/VA 블록용 패턴만 따로 떼기 (blk 0~3) ===
        vmva_mask = (self.jac_blk_type >= 0) & (self.jac_blk_type <= 3)

        self.jac_rows_vmva_single = self.jac_rows_single[vmva_mask]
        self.jac_cols_vmva_single = self.jac_cols_single[vmva_mask]
        self.jac_blk_type_vmva    = self.jac_blk_type[vmva_mask]
        self.jac_bus_i_vmva       = self.jac_bus_i[vmva_mask]
        self.jac_bus_j_vmva       = self.jac_bus_j[vmva_mask]
        self.jac_G_ij_vmva        = self.jac_G_ij[vmva_mask]
        self.jac_B_ij_vmva        = self.jac_B_ij[vmva_mask]

        # vmva 패턴이 big_vals 안에서 차지하는 위치 인덱스 (0..nnz_single-1 중 subset)
        all_idx = torch.arange(self.jac_nnz_single, dtype=torch.long)
        self.jac_vmva_pos_single = all_idx[vmva_mask]  # len = nnz_vmva

        self.jac_nnz_vmva_single = self.jac_rows_vmva_single.numel()

        # === 8) (blk_type, bus_i, bus_j) -> k 룩업 (디버그/검증용) ===
        # (원래 코드 그대로)
        key2k = {}
        bt = self.jac_blk_type.cpu().numpy().astype(np.int64)
        bi = self.jac_bus_i.cpu().numpy().astype(np.int64)
        bj = self.jac_bus_j.cpu().numpy().astype(np.int64)
        for k in range(int(self.jac_nnz_single)):
            key2k[(int(bt[k]), int(bi[k]), int(bj[k]))] = k
        self._key2k = key2k


    ### 아래 kernel 에 대한 설명
    ### 지금 b 와 k가 값이 1블럭당 값이 채워져 있는 것에 대해서. 즉, 지금은 10582. *5 = 52910개. 물론, 여기에는 0 값도 있어야 하는 것이 인지상정.
    ### 이때, b는 0~4까지, k는 0~10581까지. 즉, b가 0일때 k가 0~10581까지 다 채워지고, b가 1일때 k가 0~10581까지 다 채워지고, 이런식으로 되어 있음.
    ### blk는 0 -> P_VM, 1 -> P_VA, 2 -> Q_VM, 3 -> Q_VA, 4 -> P_PG, 5 -> Q_QG
    ### 특히, 1,2 2ng개가 각각 4, 5로 채워지고 나서 이후 전압 블록이 (i,j)를 (i우선, j다음)순으로 사전 정렬한 ij순서대로 각 (i,j)마다 
    ### P_VM, P_VA, Q_VM, Q_VA가 차례대로 채워지는 형태임. 따라서, 0, 1, 2, 3이 각각 반복. 
    ### P_PG와 Q_QG는 1로 이루어지게 되므로 따로 kernel에 포함 x 
    ### 이때, 각 value의 순서를 잘 생각해보면.. i먼저, j다음이므로 잘 만들어졌다는 가정하에 생각을 하면
    ### 각 배치당 가장 첫 줄 ~ 다음 줄 ~ ... 즉, 각 블럭의 자코비언의 순서와 들어맞는다. 
    
    def _compile_fill_kernel(self):
        code = r'''
        extern "C" __global__
        void fill_vm_va_blocks_2d(
            const int B,
            const int nbus,
            const int nnz_single,   // full single-block nnz (PG/QG 포함)
            const int nnz_vmva,     // VM/VA 엔트리 개수
            const int*  __restrict__ pos,      // length nnz_vmva, full index 위치
            const signed char* __restrict__ blk_type, // 0~3만
            const int*  __restrict__ bus_i,
            const int*  __restrict__ bus_j,
            const float* __restrict__ Gij,
            const float* __restrict__ Bij,
            const float* __restrict__ vm,    // (B*nbus)
            const float* __restrict__ va,    // (B*nbus)
            const float* __restrict__ vr,    // (B*nbus)
            const float* __restrict__ vi,    // (B*nbus)
            const float* __restrict__ cosv,  // (B*nbus)
            const float* __restrict__ sinv,  // (B*nbus)
            const float* __restrict__ Ai,    // (B*nbus)
            const float* __restrict__ Bi,    // (B*nbus)
            float* __restrict__ out_vals     // (B*nnz_single)
        ){
            int kk = blockIdx.x * blockDim.x + threadIdx.x; // vmva 인덱스
            if (kk >= nnz_vmva) return;

            int b = blockIdx.y;
            if (b >= B) return;

            // VM/VA 전용 패턴 index
            int k = kk;

            signed char bt = blk_type[k];  // 0~3만 온다고 가정
            int i = bus_i[k];
            int j = bus_j[k];

            int base     = b * nbus;
            int out_base = b * nnz_single;

            float vr_i  = vr  [base + i];
            float vi_i  = vi  [base + i];
            float cos_i = cosv[base + i];
            float sin_i = sinv[base + i];
            float Ai_i  = Ai  [base + i];
            float Bi_i  = Bi  [base + i];

            float cos_j = cosv[base + j];
            float sin_j = sinv[base + j];

            float G  = Gij[k];
            float Bc = Bij[k];

            float fdiag = (i == j) ? 1.0f : 0.0f;

            float t = 0.f;

            if (bt == 0) {
                // P_VM
                t += fdiag * (-(cos_i * Ai_i + sin_i * Bi_i));
                t += - vr_i * ( G * cos_j - Bc * sin_j );
                t += - vi_i * ( Bc * cos_j + G  * sin_j );
            }
            else if (bt == 1) {
                // P_VA
                float vr_j = vr[base + j];
                float vi_j = vi[base + j];
                t += fdiag * ( vi_i * Ai_i - vr_i * Bi_i );
                t += vr_i * ( G  * vi_j + Bc * vr_j );
                t += vi_i * ( Bc * vi_j - G  * vr_j );
            }
            else if (bt == 2) {
                // Q_VM
                t += fdiag * ( cos_i * Bi_i - sin_i * Ai_i );
                t += vr_i * ( Bc * cos_j + G  * sin_j );
                t += -vi_i * ( G  * cos_j - Bc * sin_j );
            }
            else { // bt == 3
                // Q_VA
                float vr_j = vr[base + j];
                float vi_j = vi[base + j];
                t += fdiag * ( -vi_i * Bi_i - vr_i * Ai_i );
                t += vr_i * ( Bc * (-vi_j) + G  * vr_j );
                t += -vi_i * ( G  * (-vi_j) - Bc * vr_j );
            }

            // VM/VA 엔트리가 full big_vals 안에서 차지하는 위치
            int full_idx = pos[k];                 // 0..nnz_single-1
            out_vals[out_base + full_idx] = t;     // b 배치의 해당 위치에 쓰기
        }
        ''';
        self._fill_kernel = cp.RawKernel(code, 'fill_vm_va_blocks_2d')




    def _compile_coo_slice_blocks_kernel(self):
        code = r'''
        extern "C" __global__
        void coo_slice_blocks(
            // input COO (global indices)
            const long long NNZ_in,
            const long long B,
            const long long m,         // block row size
            const long long n,         // block col size
            const long long* __restrict__ big_rows, // len=NNZ_in (global row)
            const long long* __restrict__ big_cols, // len=NNZ_in (global col)
            const float*     __restrict__ big_vals, // len=NNZ_in

            // local slice maps for ONE block (shared across all B blocks)
            // row_map[r_loc] = [0..R-1] if selected else -1
            // col_map[c_loc] = [0..C-1] if selected else -1
            const int*  __restrict__ row_map, // len=m
            const int*  __restrict__ col_map, // len=n
            const int   R,                    // |row_sel_local|
            const int   C,                    // |col_sel_local|

            // output buffers (over-allocated to NNZ_in)
            long long* __restrict__ out_rows, // len=NNZ_in
            long long* __restrict__ out_cols, // len=NNZ_in
            float*     __restrict__ out_vals, // len=NNZ_in

            // global counter (MUST be zeroed before launch)
            unsigned long long int* __restrict__ out_count // len=1
        ){
            long long tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= NNZ_in) return;

            // read input
            long long gr = big_rows[tid]; // global row
            long long gc = big_cols[tid]; // global col

            // decode batch and local indices: gr = b*m + r_loc, gc = b*n + c_loc
            long long b     = (m > 0) ? (gr / m) : 0;

            long long r_loc = gr - b * m;
            long long c_loc = gc - b * n;


            // map local -> sliced-local
            int rr = row_map[(int)r_loc];  // -1 if not selected
            int cc = col_map[(int)c_loc];  // -1 if not selected
            if (rr < 0 || cc < 0) return;

            // place into block-diagonal output coordinates
            // out_row = b*R + rr, out_col = b*C + cc
            long long orow = b * (long long)R + (long long)rr;
            long long ocol = b * (long long)C + (long long)cc;

            // append to output via atomic counter
            unsigned long long int pos = atomicAdd(out_count, 1ULL);
            out_rows[pos] = orow;
            out_cols[pos] = ocol;
            out_vals[pos] = big_vals[tid];
        }
        ''';
        self._coo_slice_blocks = cp.RawKernel(code, 'coo_slice_blocks')

    def prepare_big_jac_buffers(self, B: int):
        self.jac_B = int(B)
        m = int(self.jac_m_rows)
        n = int(self.jac_n_cols)
        nnz_single = int(self.jac_nnz_single)

        rows_big, cols_big = [], []
        for b in range(B):
            rows_big.append(self.jac_rows_single + b*m)
            cols_big.append(self.jac_cols_single + b*n)
        rows_big = torch.cat(rows_big, 0)
        cols_big = torch.cat(cols_big, 0)

        to_cp = lambda t: cp.asarray(t.cpu().contiguous().numpy())

        # === full 패턴 (PG/QG 포함) ===
        self.big_rows = to_cp(rows_big).astype(cp.int32)
        self.big_cols = to_cp(cols_big).astype(cp.int32)

        self.cp_blk_type = to_cp(self.jac_blk_type).astype(cp.int8)
        self.cp_bus_i    = to_cp(self.jac_bus_i).astype(cp.int64)
        self.cp_bus_j    = to_cp(self.jac_bus_j).astype(cp.int64)
        self.cp_G_ij     = to_cp(self.jac_G_ij).astype(cp.float32)
        self.cp_B_ij     = to_cp(self.jac_B_ij).astype(cp.float32)

        # PG/QG 위치
        self.single_pos_pg = cp.asarray(self.single_pos_pg_torch.cpu().numpy()).astype(cp.int64)
        self.single_pos_qg = cp.asarray(self.single_pos_qg_torch.cpu().numpy()).astype(cp.int64)

        # 값 버퍼 + 고정 COO 핸들
        self.big_vals = cp.zeros(B*nnz_single, dtype=cp.float32)
        self.J_coo = cp_coo((self.big_vals, (self.big_rows, self.big_cols)),
                            shape=(B*m, B*n))

        # === VM/VA용 패턴 (커널이 실제로 돌 대상) ===
        self.jac_nnz_vmva_single = int(self.jac_nnz_vmva_single)  # ensure python int

        self.cp_vmva_blk_type = to_cp(self.jac_blk_type_vmva).astype(cp.int8)
        self.cp_vmva_bus_i    = to_cp(self.jac_bus_i_vmva).astype(cp.int32)
        self.cp_vmva_bus_j    = to_cp(self.jac_bus_j_vmva).astype(cp.int32)
        self.cp_vmva_G_ij     = to_cp(self.jac_G_ij_vmva).astype(cp.float32)
        self.cp_vmva_B_ij     = to_cp(self.jac_B_ij_vmva).astype(cp.float32)

        # VM/VA 엔트리가 big_vals 내에서 가지는 위치
        self.cp_vmva_pos = to_cp(self.jac_vmva_pos_single).astype(cp.int32)

        self._compile_fill_kernel()
        self._compile_coo_slice_blocks_kernel()

        self.slice_capacity = 0
        self.slice_rows_ws  = None
        self.slice_cols_ws  = None
        self.slice_vals_ws  = None
        self.slice_count_ws = None
        self.slice_cache = {}

    ### 구조 패턴을 유지한채 내부의 data(값)만 갱신하도록 하는 함수 (커널 호출 포함) ###
    def update_big_jac_values(self, vm: torch.Tensor, va: torch.Tensor):
        B, nbus = vm.shape

        to_cp_dl = lambda t: cp.from_dlpack(t.contiguous())
        vm_cp = to_cp_dl(vm)
        va_cp = to_cp_dl(va)

        cosv = cp.cos(va_cp); sinv = cp.sin(va_cp)
        vr   = vm_cp * cosv
        vi   = vm_cp * sinv

        A  = vr @ self.Yr_dense + (-vi) @ self.Yi_dense
        Bv = vr @ self.Yi_dense +   vi  @ self.Yr_dense

        nnz_single   = int(self.jac_nnz_single)
        nnz_vmva     = int(self.jac_nnz_vmva_single)

        # PG/QG = 1.0 세팅
        self.big_vals.fill(0)

        offsets = cp.arange(B, dtype=cp.int32) * cp.int32(nnz_single)
        pg_idx  = (offsets[:, None] + self.single_pos_pg[None, :]).ravel()
        qg_idx  = (offsets[:, None] + self.single_pos_qg[None, :]).ravel()
        self.big_vals[pg_idx] = cp.float32(1.0)
        self.big_vals[qg_idx] = cp.float32(1.0)

        # --- VM/VA 패턴용 인자 준비 ---
        blk_type = self.cp_vmva_blk_type   # 0~3만
        bus_i    = self.cp_vmva_bus_i
        bus_j    = self.cp_vmva_bus_j
        G_ij     = self.cp_vmva_G_ij
        B_ij     = self.cp_vmva_B_ij
        pos      = self.cp_vmva_pos       # full big_vals 내 위치
        ### 아래 값은 수정될 수 있음. -> profiling에 근거하여 ###
        threads = 256
        grid_x  = (nnz_vmva + threads - 1) // threads
        grid_y  = int(B)

        self._fill_kernel(
            (grid_x, grid_y, 1),
            (threads, 1, 1),
            (
                np.int32(B),
                np.int32(nbus),
                np.int32(nnz_single),    # full nnz
                np.int32(nnz_vmva),      # vm/va nnz
                pos,
                blk_type,
                bus_i,
                bus_j,
                G_ij,
                B_ij,
                vm_cp.ravel(order="C"),
                va_cp.ravel(order="C"),
                vr.ravel(order="C"),
                vi.ravel(order="C"),
                cosv.ravel(order="C"),
                sinv.ravel(order="C"),
                A.ravel(order="C"),
                Bv.ravel(order="C"),
                self.big_vals,
            )
        )
        ### 스레드들간의 sinc를 맞춤 ###
        cp.cuda.Stream.null.synchronize()



    ### 블록 대각선 행렬에서 특정 블록 행/열 인덱스만 선택하여 슬라이스하는 커널 호출 함수(제일 큰 단위) ###
    ### 구조를 유지한 채 값만 바꾸어 slicing 할 수 있도록 제공 ###
    
    def _ensure_slice_workspace(self, nnz_in: int):

        if self.slice_capacity < nnz_in or self.slice_rows_ws is None:
            self.slice_rows_ws  = cp.empty((nnz_in,), dtype=cp.int64)
            self.slice_cols_ws  = cp.empty((nnz_in,), dtype=cp.int64)
            self.slice_vals_ws  = cp.empty((nnz_in,), dtype=cp.float32)
            self.slice_count_ws = cp.zeros((1,), dtype=cp.int64)
            self.slice_capacity = nnz_in
        else:
            # 매 호출 때는 카운터만 0으로 리셋(배열 전체를 0으로 만들 필요 없음)
            self.slice_count_ws.fill(0)

    def coo_slice_blocks(
        self,
        big_rows, big_cols, big_vals,     # cp.ndarray (NNZ_in,)
        B, m, n,                          # int: 블록 수, 각 블록의 행/열 크기
        row_sel_local, col_sel_local      # cp.ndarray[int64]: 각 블록의 로컬 row/col 선택 인덱스
    ):

        NNZ_in = int(big_vals.size)
        self._ensure_slice_workspace(NNZ_in)

        # 로컬 선택 인덱스를 매핑 테이블로 변환 (-1=제외, 0..R-1=채택)
        row_map = cp.full((m,), -1, dtype=cp.int32)
        col_map = cp.full((n,), -1, dtype=cp.int32)
        row_map[row_sel_local] = cp.arange(row_sel_local.size, dtype=cp.int32)
        col_map[col_sel_local] = cp.arange(col_sel_local.size, dtype=cp.int32)
        R = int(row_sel_local.size)
        C = int(col_sel_local.size)

        # 커널 실행
        ### 이 아래의 값도 마찬가지로 수정될 수 있음 ###
        threads = 256
        blocks  = (NNZ_in + threads - 1) // threads
        self._coo_slice_blocks(
            (blocks,), (threads,),
            (
                cp.int64(NNZ_in),
                cp.int64(B), cp.int64(m), cp.int64(n),
                big_rows, big_cols, big_vals,
                row_map, col_map, cp.int32(R), cp.int32(C),
                self.slice_rows_ws, self.slice_cols_ws, self.slice_vals_ws, self.slice_count_ws
            )
        )

        # 실제 출력 nnz
        nnz_out = int(self.slice_count_ws.get()[0])

        # 필요하면 여기서 (row, col) 기준 정렬 가능
        # idx = cp.lexsort((self.slice_cols_ws[:nnz_out], self.slice_rows_ws[:nnz_out]))
        # return self.slice_rows_ws[idx], self.slice_cols_ws[idx], self.slice_vals_ws[idx], R, C

        # 정렬이 필요 없으면 바로 반환(뷰)
        return (self.slice_rows_ws[:nnz_out],
                self.slice_cols_ws[:nnz_out],
                self.slice_vals_ws[:nnz_out],
                R, C)
    ### 이 부분은 배치 크기에 따른 것. 즉, 배치가 달라지면 구조를 새로 만들 수 있도록 함. ###
    ### 이로써 train 뿐 아니라 test도 가능 ###
    def ensure_batch(self, B: int):
        B = int(B)
        if self.B != B:
            self.B = B


        # 배치 의존 버퍼/핸들 준비 (B마다 1회)
        if B not in self._big_jac_ready_B:
            self.prepare_big_jac_buffers(B)   # ← 여기서 block-diag 크기(B*nbus 등)만큼 할당
            self._big_jac_ready_B.add(B)

    ######################################################################
    ### 일단 여기까지 가속기의 메인구조 ###



    
    # def ineq_jac(self, Y, cosva, sinva, vr, vi, box_const_jac_dense):
    #     #_, _, vm, va = self.get_yvars(Y)
    #     #box_const_jac_dense = box_const_jac.to_dense()
        
    #     jac = box_const_jac_dense

    #     return jac

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
        Y_partial[:, self.pg_pv_zidx] = Z[:, self.pg_pv_zidx] * self.pmax[gen_pv_idx].to(device=self.device, dtype=torch.get_default_dtype()) + \
             (1-Z[:, self.pg_pv_zidx]) * self.pmin[gen_pv_idx].to(device=self.device, dtype=torch.get_default_dtype())

        # Y_partial[:, self.pg_pv_zidx] = Z[:, self.pg_pv_zidx] * self.pmax[1:].to(dtype=torch.get_default_dtype()) + \
        #      (1-Z[:, self.pg_pv_zidx]) * self.pmin[1:].to(dtype=torch.get_default_dtype())
        
        # Re-scale real parts of voltages
        Y_partial[:, self.vm_spv_zidx] = Z[:, self.vm_spv_zidx] * self.vmax[self.spv].to(device=self.device, dtype=torch.get_default_dtype()) + \
            (1-Z[:, self.vm_spv_zidx]) * self.vmin[self.spv].to(device=self.device, dtype=torch.get_default_dtype())
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


### By 수호 : 새로운 구조체 선언 ###
### 이 아래는 solver를 관리하며 CuPy와 cuDSS간의 데이터를 바인딩 해주는 부분 ### cupyx big csr -> cuDSS big csr
### 중간 dense torch.tensor 없이 cupy csr로 바로 빌드.
######################################################################

class PersistentNewtoncsrcudss:
    def __init__(self, reorder_alg=1, fix_pattern=True, verify_every=0):
        self.cs = cs 

        # forward (A=J) 핸들/버퍼
        self.h_f = None; self.cfg_f = None; self.data_f = None; self.A_f = None
        self.Bdn_f = None; self.Xdn_f = None
        self.indptr_t_f = None; self.indices_t_f = None; self.values_t_f = None; self.b_t_f = None
        self.indptr_cp_f = None; self.indices_cp_f = None; self.values_cp_f = None; self.b_cp_f = None; self.x_cp_f = None
        self.meta_f = None

        # backward (A=J^T) 핸들/버퍼
        self.h_bt = None; self.cfg_bt = None; self.data_bt = None; self.A_bt = None
        self.Bdn_bt = None; self.Xdn_bt = None
        self.indptr_t_bt = None; self.indices_t_bt = None; self.values_t_bt = None; self.b_t_bt = None
        self.indptr_cp_bt = None; self.indices_cp_bt = None; self.values_cp_bt = None; self.b_cp_bt = None; self.x_cp_bt = None
        self.meta_bt = None

        # 공통 메타
        self.B = None; self.m = None; self.N = None
        self.k = None; self.nnz_total = None
        self.reorder_alg = int(reorder_alg)
        self.fix_pattern = bool(fix_pattern)
        self.verify_every = int(verify_every)  # 0이면 아예 검증 안 함
        self._step = 0
        self._pattern_hash = None  # 초기에만 설정

    @staticmethod
    def _pattern_key_cupy(J_csr_cp) -> int:
        # indptr/indices 바이트를 해싱 → 패턴 고유키
        indptr_bytes  = J_csr_cp.indptr.view(cp.uint8).get().tobytes()
        indices_bytes = J_csr_cp.indices.view(cp.uint8).get().tobytes()
        return hash(indptr_bytes + indices_bytes)
    @torch.no_grad()
    def _build_from_cupy_csr(self, J_csr_cp, F_cp_1d, *, for_transpose: bool):
        cs = self.cs
        N = int(J_csr_cp.shape[0])
        indptr_cp, indices_cp, values_cp = J_csr_cp.indptr, J_csr_cp.indices, J_csr_cp.data
        nnz = int(indices_cp.size)

        h = cs.create(); cfg = cs.config_create(); data = cs.data_create(h)
        alg_val = np.array([int(self.reorder_alg)], dtype=np.int32)
        cs.config_set(cfg, int(cs.ConfigParam.REORDERING_ALG), alg_val.ctypes.data, alg_val.nbytes)

        A = cs.matrix_create_csr(
            N, N, nnz,
            int(indptr_cp.data.ptr), int(indptr_cp[1:].data.ptr),
            int(indices_cp.data.ptr), int(values_cp.data.ptr),
            int(CD.CUDA_R_32I), int(CD.CUDA_R_32F),
            int(cs.MatrixType.GENERAL),
            int(cs.MatrixViewType.FULL),
            int(cs.IndexBase.ZERO)
        )

        # RHS/솔루션 버퍼
        b_cp = cp.from_dlpack(F_cp_1d)
        x_cp = cp.zeros((N,1), dtype=cp.float32)
        Bdn = cs.matrix_create_dn(N,1,N,int(b_cp.data.ptr), int(CD.CUDA_R_32F), int(cs.Layout.COL_MAJOR))
        Xdn = cs.matrix_create_dn(N,1,N,int(x_cp.data.ptr), int(CD.CUDA_R_32F), int(cs.Layout.COL_MAJOR))

        cp.cuda.runtime.deviceSynchronize()
        cs.execute(h, cs.Phase.ANALYSIS,      cfg, data, A, Xdn, Bdn)
        cs.execute(h, cs.Phase.FACTORIZATION, cfg, data, A, Xdn, Bdn)

        # 어느 슬롯에 저장할지 분기만 다름 (나머지 로직은 동일)
        if not for_transpose:
            self.h_f, self.cfg_f, self.data_f, self.A_f = h, cfg, data, A
            self.Bdn_f, self.Xdn_f = Bdn, Xdn
            self.indptr_cp_f, self.indices_cp_f, self.values_cp_f = indptr_cp, indices_cp, values_cp
            self.b_cp_f, self.x_cp_f = b_cp, x_cp
        else:
            self.h_bt, self.cfg_bt, self.data_bt, self.A_bt = h, cfg, data, A
            self.Bdn_bt, self.Xdn_bt = Bdn, Xdn
            self.indptr_cp_bt, self.indices_cp_bt, self.values_cp_bt = indptr_cp, indices_cp, values_cp
            self.b_cp_bt, self.x_cp_bt = b_cp, x_cp
    @torch.no_grad()
    def ensure_built(self, J_csr_cp, F_cp_1d, *, Jt_csr_cp=None, use_backward=True):
        """
        J_csr_cp : cupyx.scipy.sparse.csr_matrix (N×N, N=B*m)
        F_cp_1d  : cupy.ndarray (N,) float32 (처음 빌드시만 필요, 이후 solve에서 갱신)
        Jt_csr_cp: (선택) CSR(J^T). 없으면 처음 1회 transpose().tocsr()로 생성
        use_backward: True면 J^T 경로도 원래처럼 함께 빌드
        """

        N = int(J_csr_cp.shape[0])

        need_build = (self.h_f is None) or (self.N != N)
        if need_build:
            # forward(J) 한 번
            self._build_from_cupy_csr(J_csr_cp, F_cp_1d, for_transpose=False)

            # backward(J^T)도 원래처럼 한 번 (원치 않으면 use_backward=False)
            if use_backward:
                if Jt_csr_cp is None:
                    Jt_csr_cp = (J_csr_cp.transpose()).tocsr()
                self._build_from_cupy_csr(Jt_csr_cp, F_cp_1d, for_transpose=True)

            self.N = N
            self.nnz_total = int(J_csr_cp.indices.size)


    
    # ---------- per step ----------
    @torch.no_grad()
    def solve_forward(self, J_csr_cp, F_cp_1d):
        cs = self.cs
        self.values_cp_f[...] = J_csr_cp.data
        self.b_cp_f[...] = cp.from_dlpack(F_cp_1d)
        cp.cuda.runtime.deviceSynchronize()
        cs.execute(self.h_f, cs.Phase.REFACTORIZATION, self.cfg_f, self.data_f, self.A_f, self.Xdn_f, self.Bdn_f)
        cs.execute(self.h_f, cs.Phase.SOLVE,           self.cfg_f, self.data_f, self.A_f, self.Xdn_f, self.Bdn_f)
        ## 반환할 때는 그냥 torch 로. 외부에서 다시 Batch로 reshape!
        return torch.from_dlpack(self.x_cp_f)
    @torch.no_grad()
    def solve_backward(self, Jt_csr_cp, F_cp_1d):
        cs = self.cs
        self.values_cp_bt[...] = Jt_csr_cp.data
        self.b_cp_bt[...] = cp.from_dlpack(F_cp_1d)
        cp.cuda.runtime.deviceSynchronize()
        cs.execute(self.h_bt, cs.Phase.REFACTORIZATION, self.cfg_bt, self.data_bt, self.A_bt, self.Xdn_bt, self.Bdn_bt)
        cs.execute(self.h_bt, cs.Phase.SOLVE,           self.cfg_bt, self.data_bt, self.A_bt, self.Xdn_bt, self.Bdn_bt)
        return torch.from_dlpack(self.x_cp_bt)

    def destroy(self):
        cs = self.cs
        for h, cfg, data, A, Bdn, Xdn in [
            (self.h_f, self.cfg_f, self.data_f, self.A_f, self.Bdn_f, self.Xdn_f),
            (self.h_bt, self.cfg_bt, self.data_bt, self.A_bt, self.Bdn_bt, self.Xdn_bt),
        ]:
            if A is not None:
                cs.matrix_destroy(A); cs.matrix_destroy(Bdn); cs.matrix_destroy(Xdn)
        if self.data_f is not None: cs.data_destroy(self.h_f, self.data_f)
        if self.data_bt is not None: cs.data_destroy(self.h_bt, self.data_bt)
        if self.cfg_f is not None: cs.config_destroy(self.cfg_f)
        if self.cfg_bt is not None: cs.config_destroy(self.cfg_bt)
        if self.h_f is not None: cs.destroy(self.h_f)
        if self.h_bt is not None: cs.destroy(self.h_bt)
        self.__init__(self.reorder_alg)  # reset

### 이 아래는 coo -> csr 패턴/매핑 관리 클래스 ###


class Coo2CsrPatternGPU:
    """
    - ensure_built(...) 호출 시:
        * 현재 (row, col, n_rows, n_cols, B) 정보와
          기존 패턴이 같은지 검사.
        * 다르면 reset() 후 CSR 패턴 새로 build.
        * 같으면 아무 것도 안 함 (기존 구조 재사용).
    - update_from_coo(...) :
        * 새로운 COO data만 받아서 self.csr.data만 갱신.
    """

    def __init__(self):
        self._built = False
        self.csr = None

        # 패턴 메타 정보
        self.n_rows = None
        self.n_cols = None
        self.nnz_coo = None
        self.B = None
        self.dtype = None

        # row/col 패턴 signature
        self._sig_row = None
        self._sig_col = None

        # CSR 패턴 → key 기반 매핑용
        self._key_csr_sorted = None
        self._order = None

    def reset(self):
        self._built = False
        self.csr = None

        self.n_rows = None
        self.n_cols = None
        self.nnz_coo = None
        self.B = None
        self.dtype = None

        self._sig_row = None
        self._sig_col = None

        self._key_csr_sorted = None
        self._order = None

    def _need_rebuild(self, row, col, n_rows, n_cols, B):
        if not self._built:
            return True

        if (n_rows != self.n_rows) or (n_cols != self.n_cols):
            return True

        if (self.B is not None) and (B is not None) and (B != self.B):
            return True

        if row.size != self.nnz_coo:
            return True

        sig_row = int(cp.sum(row).item())
        sig_col = int(cp.sum(col).item())

        if (self._sig_row != sig_row) or (self._sig_col != sig_col):
            return True

        return False

    def ensure_built(self, row, col, data, n_rows, n_cols, B=None):
        row = cp.asarray(row, dtype=cp.int64)
        col = cp.asarray(col, dtype=cp.int64)
        data = cp.asarray(data)

        n_rows = int(n_rows)
        n_cols = int(n_cols)

        if not self._need_rebuild(row, col, n_rows, n_cols, B):
            return

        self.reset()

        self.dtype = data.dtype
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.nnz_coo = int(row.size)
        self.B = B

        self._sig_row = int(cp.sum(row).item())
        self._sig_col = int(cp.sum(col).item())

        csr_ref = cp_csr((data, (row, col)),
                             shape=(self.n_rows, self.n_cols))

        indices = csr_ref.indices.copy()
        indptr  = csr_ref.indptr.copy()
        data0   = cp.zeros_like(csr_ref.data, dtype=self.dtype)

        self.csr = cp_csr((data0, indices, indptr),
                              shape=csr_ref.shape)

        self.nnz_csr = int(self.csr.data.size)

        k = cp.arange(self.nnz_csr, dtype=cp.int64)
        row_csr = cp.searchsorted(indptr, k, side="right") - 1
        col_csr = indices

        n_cols64 = cp.int64(self.n_cols)
        key_csr = row_csr * n_cols64 + col_csr

        self._order = cp.argsort(key_csr)
        self._key_csr_sorted = key_csr[self._order]

        self._built = True

    def update_from_coo(self, row, col, data):
        assert self._built

        row = cp.asarray(row, dtype=cp.int64)
        col = cp.asarray(col, dtype=cp.int64)
        data = cp.asarray(data, dtype=self.dtype)

        assert row.size == col.size == data.size

        n_cols64 = cp.int64(self.n_cols)
        key_coo = row * n_cols64 + col

        pos = cp.searchsorted(self._key_csr_sorted, key_coo)
        csr_idx = self._order[pos]

        buf = self.csr.data
        buf[...] = 0
        buf[:] = cp.bincount(csr_idx,
                             weights=data,
                             minlength=self.nnz_csr)

######################################################################
### 여기까지가 새로운 구조체 선언 ###

def PFFunction(data, tol=1e-2, bsz=50, max_iters=5):  # bsz를 꼭 지금 설정한 batch size와 맞추기!!
    solver2 = get_solver_for_current_device2(reorder_alg=1, fix_pattern=True)

    class PFFunctionFn(Function):
        @staticmethod
        def forward(ctx, X, Z):
            # nvtx.range_push("PFFunction Forward")
            # try:
            #     with record_function("PF_forward_total"):
            # forward_time = time.perf_counter()

            ### 주석처리한 것은 CUDA profiling을 위한 부분. 무시 ###
###                    with record_function("PF_forward_init_Y"):
            Y = torch.zeros(X.shape[0], data.ydim, device=DEVICE)

            # known/estimated values (pg at pv buses, vm at all gens, va at slack bus)
            Y[:, data.pg_start_yidx + data.pv_] = Z[:, data.pg_pv_zidx]    # pg at non-slack gens
            Y[:, data.vm_start_yidx + data.spv] = Z[:, data.vm_spv_zidx]   # vm at gens
            Y[:, data.va_start_yidx + data.slack] = torch.tensor(
                data.slack_va,
                device=DEVICE,
                dtype=torch.get_default_dtype()
            )  # va at slack bus

            # init guesses for remaining values
            Y[:, data.vm_start_yidx + data.pq] = torch.tensor(
                data.vm_init[data.pq],
                device=DEVICE,
                dtype=torch.get_default_dtype()
            )  # vm at load buses
            Y[:, data.va_start_yidx + data.pv] = torch.tensor(
                data.va_init[data.pv],
                device=DEVICE,
                dtype=torch.get_default_dtype()
            )  # va at non-slack gens
            Y[:, data.va_start_yidx + data.pq] = torch.tensor(
                data.va_init[data.pq],
                device=DEVICE,
                dtype=torch.get_default_dtype()
            )  # va at load buses
            Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] = 0    # qg at gens (not used in Newton upd)
            Y[:, data.pg_start_yidx + data.slack_] = 0                    # pg at slack (not used in Newton upd)

            last_eqs = np.concatenate([data.pflow_start_eqidx + data.slack,
                                        data.qflow_start_eqidx + data.spv])
            last_vars = np.concatenate([
                data.pg_start_yidx + data.slack_,
                np.arange(data.qg_start_yidx, data.qg_start_yidx + data.ng)
            ])
            n_block = 2 * data.nbus + 2 * data.ng
            vm_start_yidx_slice = np.arange(int(data.vm_start_yidx),
                                            int(n_block),
                                            dtype=np.int64)

            keep_constr = np.concatenate([
                data.pflow_start_eqidx + data.pv,     # real power flow at non-slack gens
                data.pflow_start_eqidx + data.pq,     # real power flow at load buses
                data.qflow_start_eqidx + data.pq
            ])
            newton_guess_inds = np.concatenate([
                data.vm_start_yidx + data.pq,         # vm at load buses
                data.va_start_yidx + data.pv,         # va at non-slack gens
                data.va_start_yidx + data.pq          # va at load buses
            ])

            m = 2 * data.nbus
            n = 2 * data.nbus + 2 * data.ng
            mr = len(keep_constr)
            mc = len(newton_guess_inds)
            pattern_J_red = data.pattern_J_red

            # -------------------- (2) Newton 반복 (batch loop 포함) --------------------
            # with record_function("PF_forward_newton_loops"):
            for b in range(0, X.shape[0], bsz):
                X_b = X[b:b+bsz]
                Y_b = Y[b:b+bsz]
                data.ensure_batch(Y_b.shape[0])
                M = data.B * mr
                N = data.B * mc

                for it in range(max_iters):
                    # ---- (2-1) 잔차 계산 eq_resid ----
                    # with record_function("PF_forward_eq_resid"):
                    #pre_t = time.perf_counter()
                    gy = data.eq_resid(X_b, Y_b)[:, keep_constr]
                    #pre_process_time = time.perf_counter() - pre_t
                    # 필요하면 debug print:
                    # print(f"[FW] eq_resid time: {pre_process_time:.6f}s")

                # ---- (2-2) big Jacobian 값 업데이트 ----
                # with record_function("PF_forward_update_big_jac_values"):
                    #jac_t = time.perf_counter()
                    _, _, vm, va = data.get_yvars(Y_b)
                    data.update_big_jac_values(vm, va)
                    J_coo = data.J_coo
                    J_coo.data = data.big_vals
                    #jacobian_building_time = time.perf_counter() - jac_t
                    # print(f"[FW] update_big_jac_values: {jacobian_building_time:.6f}s")

                    # 이 시점의 full J를 ctx에 저장할 수 있도록 보관
                    J_check = J_coo
                    J_r = J_check.row.astype(cp.int64, copy=True)
                    J_c = J_check.col.astype(cp.int64, copy=True)
                    J_v = J_check.data.astype(cp.float32, copy=True)

                # ---- (2-3) coo_slice_blocks + 패턴/CSR 구성 ----
                # with record_function("PF_forward_slice_and_csr"):
                    #slice_t = time.perf_counter()
                    out_r, out_c, out_v, R, C = data.coo_slice_blocks(
                        J_r, J_c, J_v,
                        data.B, m, n,
                        keep_constr, newton_guess_inds
                    )

                    pattern_J_red.ensure_built(out_r, out_c, out_v, M, N, B=data.B)
                    pattern_J_red.update_from_coo(out_r, out_c, out_v)
                    J_red_csr = pattern_J_red.csr
                    #slicing_time = time.perf_counter() - slice_t
                    # print(f"[FW] slicing+CSR time: {slicing_time:.6f}s")

                # ---- (2-4) 선형 시스템 풀이 (cuDSS) ----
                # with record_function("PF_forward_linear_solve"):
                    #lin_t = time.perf_counter()
                    solver2.ensure_built(J_red_csr, gy, use_backward=True)  # cuDSS build (패턴 고정)
                    delta = solver2.solve_forward(J_red_csr, gy)
                    delta = delta.view(gy.shape[0], -1).contiguous()
                    #cp.cuda.Stream.null.synchronize()
                    #linear_solve_time = time.perf_counter() - lin_t
                    # print(f"[FW] linear_solve time: {linear_solve_time:.6f}s")

                # ---- (2-5) Newton 업데이트 및 수렴 체크 ----
                # with record_function("PF_forward_update_solution"):
                    Y_b[:, newton_guess_inds] -= delta
                    if delta.abs().max(dim=1).values.mean() < tol:
                        break

            # -------------------- (3) Post-process: qg, slack pg 계산 --------------------
            # with record_function("PF_forward_post_process"):
            Y[:, data.qg_start_yidx:data.qg_start_yidx + data.ng] = \
                -data.eq_resid(X, Y)[:, data.qflow_start_eqidx + data.spv]
            Y[:, data.pg_start_yidx + data.slack_] = \
                -data.eq_resid(X, Y)[:, data.pflow_start_eqidx + data.slack]

            # -------------------- (4) backward에서 쓸 컨텍스트 저장 --------------------
            # with record_function("PF_forward_save_ctx"):
            ctx.data = data
            ctx.J_r = J_r
            ctx.J_c = J_c
            ctx.J_v = J_v
            ctx.M = M
            ctx.N = N
            ctx.out_r = out_r
            ctx.out_c = out_c
            ctx.out_v = out_v
            ctx.B_block = int(data.B)
            ctx.m_block = int(2 * data.nbus)
            ctx.n_block = int(2 * data.nbus + 2 * data.ng)
            ctx.last_eqs = last_eqs
            ctx.last_vars = last_vars
            ctx.keep_constr = keep_constr
            ctx.vm_start_yidx_slice = vm_start_yidx_slice
            ctx.newton_guess_inds = newton_guess_inds

            #forward_time = time.perf_counter() - forward_time
            # print(f"[FW] Total forward time: {forward_time:.6f}s")

            return Y
            # finally:
            #     nvtx.range_pop()

        @staticmethod
        def backward(ctx, dl_dy):
            # nvtx.range_push("PFFunction Backward")
            # try:
            #     with record_function("PF_backward_total"):
                    # backward_time = time.perf_counter()
            data = ctx.data

            # -------------------- (1) ctx에서 기본 정보 꺼내기 --------------------
            # with record_function("PF_backward_ctx_unpack"):
            J_r = ctx.J_r
            J_c = ctx.J_c
            J_v = ctx.J_v
            out_r = ctx.out_r
            out_c = ctx.out_c
            out_v = ctx.out_v
            M = ctx.M
            N = ctx.N
            B_block = ctx.B_block
            m_block = ctx.m_block
            n_block = ctx.n_block
            last_eqs = ctx.last_eqs
            last_vars = ctx.last_vars
            keep_constr = ctx.keep_constr
            vm_start_yidx_slice = ctx.vm_start_yidx_slice
            newton_guess_inds = ctx.newton_guess_inds
            pattern_J_red_T = data.pattern_J_red_T
            pattern_J3_T = data.pattern_J3_T
            pattern_J2_T = data.pattern_J2_T

            # -------------------- (2) J_check 재구성 --------------------
            # with record_function("PF_backward_build_J_from_ctx"):
            # J_from_ctx_time = time.perf_counter()
            J_check = cp_coo((J_v, (J_r, J_c)),
                                shape=(B_block * m_block, B_block * n_block))
            # cp.cuda.Stream.null.synchronize()
            # J_from_ctx_time = time.perf_counter() - J_from_ctx_time
            # print(f"[BW] J_from_ctx_time: {J_from_ctx_time:.6f}s")

        # -------------------- (3) J_red^T (Newton jacobian) 구성 --------------------
            # with record_function("PF_backward_build_Jred_T"):
            pattern_J_red_T.ensure_built(out_c, out_r, out_v, N, M, B=data.B)
            pattern_J_red_T.update_from_coo(out_c, out_r, out_v)
            J_red_csr_T = pattern_J_red_T.csr

            # -------------------- (4) J3, J2 slice 및 CSR 구성 --------------------
            # with record_function("PF_backward_slice_J3_build"):
            # slicing1_time = time.perf_counter()
            out_r3, out_c3, out_v3, R, C = data.coo_slice_blocks(
                J_check.row.astype(cp.int64, copy=False),
                J_check.col.astype(cp.int64, copy=False),
                J_check.data.astype(cp.float32, copy=False),
                data.B, m_block, n_block,
                last_eqs, vm_start_yidx_slice,
            )
            # slicing1_time = time.perf_counter() - slicing1_time

            M3 = data.B * (2 * data.nbus)
            N3 = data.B * len(last_eqs)
            # make_J_and_T_csr_time = time.perf_counter()

            pattern_J3_T.ensure_built(out_c3, out_r3, out_v3, M3, N3, B=data.B)
            pattern_J3_T.update_from_coo(out_c3, out_r3, out_v3)
            J_3 = pattern_J3_T.csr  # already transposed (T)

            # make_J_and_T_csr_time = time.perf_counter() - make_J_and_T_csr_time
                # print(f"[BW] slice J3: {slicing1_time:.6f}s, build J3_T: {make_J_and_T_csr_time:.6f}s")

            # with record_function("PF_backward_slice_J2_build"):
            # slicing2_time = time.perf_counter()
            out_r2, out_c2, out_v2, R, C = data.coo_slice_blocks(
                J_check.row.astype(cp.int64, copy=False),
                J_check.col.astype(cp.int64, copy=False),
                J_check.data.astype(cp.float32, copy=False),
                data.B, m_block, n_block,
                keep_constr, data.vm_start_yidx + data.spv,
            )
            # slicing2_time = time.perf_counter() - slicing2_time

            M2 = data.B * len(data.vm_start_yidx + data.spv)
            N2 = data.B * len(keep_constr)
            # make_J_and_T_csr2_time = time.perf_counter()

            pattern_J2_T.ensure_built(out_c2, out_r2, out_v2, M2, N2, B=data.B)
            pattern_J2_T.update_from_coo(out_c2, out_r2, out_v2)
            J_2 = pattern_J2_T.csr

            # make_J_and_T_csr2_time = time.perf_counter()
                # print(f"[BW] slice J2: {slicing2_time:.6f}s, build J2_T: {make_J_and_T_csr2_time:.6f}s")

            # -------------------- (5) dl_dy_3, dl_dx_3 계산 (J3·dl) --------------------
            # with record_function("PF_backward_J3_matvec_and_vec_update"):
            dl_dy_3 = torch.zeros(dl_dy.shape, device=DEVICE)
            dl_dx_3 = torch.zeros(dl_dy.shape[0], data.xdim, device=DEVICE)
            dl_dz_2 = torch.zeros(dl_dy.shape[0], data.npv + data.ng, device=DEVICE)
            dl_dx_2 = torch.zeros(dl_dy.shape[0], data.xdim, device=DEVICE)

            dl_dy_cp = cp.from_dlpack(dl_dy)
            dl_dy_cp_ravel = dl_dy_cp[:, last_vars].ravel()
            dl_dvmva_3_torch = torch.from_dlpack(
                -J_3.dot(dl_dy_cp_ravel)
            ).view(dl_dy.shape[0], -1).contiguous()

            dl_dy_3[:, data.vm_start_yidx:] = dl_dvmva_3_torch
            dl_dy_total = dl_dy_3 + dl_dy
            dl_dx_3[:, np.concatenate([data.slack, data.nbus + data.spv])] = dl_dy[:, last_vars]

            # -------------------- (6) Newton adjoint solve (J_red^T) --------------------
            # with record_function("PF_backward_linear_solve_JredT"):
            rhs = dl_dy_total[:, newton_guess_inds].contiguous().to(torch.float32)
            delta = solver2.solve_backward(J_red_csr_T, rhs)
            d_int = delta.view(rhs.shape[0], -1).contiguous()

            # -------------------- (7) J2·d_int 및 나머지 벡터 업데이트 --------------------
            # with record_function("PF_backward_J2_matvec_and_vec_update"):
            dl_dz_2[:, data.pg_pv_zidx] = -d_int[:, :data.npv]
            dl_dz_2[:, data.vm_spv_zidx] = -torch.from_dlpack(
                J_2.dot(cp.from_dlpack(d_int).ravel())
            ).view(dl_dy.shape[0], -1).contiguous()

            dl_dx_2[:, data.pv] = d_int[:, :data.npv]                       # dl_dpd at pv buses
            dl_dx_2[:, data.pq] = d_int[:, data.npv:data.npv + len(data.pq)]  # dl_dpd at pq buses
            dl_dx_2[:, data.nbus + data.pq] = d_int[:, -len(data.pq):]       # dl_dqd at pq buses

            dl_dx_total = dl_dx_3 + dl_dx_2
            dl_dz_total = dl_dz_2 + dl_dy_total[:, np.concatenate([
                data.pg_start_yidx + data.pv_, data.vm_start_yidx + data.spv
            ])]

            # torch.cuda.synchronize()
            # backward_time = time.perf_counter() - backward_time
            # print(f"[BW] Total backward time: {backward_time:.6f}s")

            return dl_dx_total, dl_dz_total
            # finally:
            #     nvtx.range_pop()

    return PFFunctionFn.apply