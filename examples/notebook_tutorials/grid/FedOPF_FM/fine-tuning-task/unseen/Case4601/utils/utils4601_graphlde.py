'''
This code is kind of improved version of DeepLDE (i.e, DeepLDE+) for large scale problems.

There are two main changes in this code.
- pytorch einsum function
- modified the terminate code in forward part in PFFunction

Moreover, when considering the large-scale problems, the batch size may need to be minimized.

* This code is also ignore the line thermal limit constraint violations.
'''
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
from scipy.sparse import csc_matrix, coo_matrix
# import cupy as cp
# import cupy.linalg

import hashlib
from copy import deepcopy
import scipy.io as spio
import time

from pypower.api import loadcase
from pypower.api import runopf, opf, makeYbus, ext2int, int2ext
from pypower import idx_bus, idx_gen, idx_brch, ppoption, runpf

# modify by kjsong
# from utils.load_pglib_opf import load_pglib_opf

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

    def __init__(self, filename, train_frac_semi = 0.0833, valid_frac=0.0833*2, test_frac=0):
        data = spio.loadmat(filename)
        self.nbus = 4601

        ## Define useful power network quantities and indices
        # ppc = CASE_FNS[self.nbus]() # NOTE: what is this???
        ppc = loadcase("./data/matpower/pglib_opf_case4601_goc.mat")
        ppc = ext2int(ppc)

        # make modified network!
        # branch flow limit of matpower = branch flow limit of pglib_opf
        # pglib_branch = load_pglib_opf(path=r"C:\Users\kjson\Desktop\DeepLDE\deeplde_skj\data\pglib_data\pglib_opf_case57_ieee.m")
        # branch = pglib_branch[2].iloc[:,5:8].values
        # ppc['branch'][:,5:8] = branch

        self.ppc = ppc

        ### kj-- Pglib와 KPX의 genbase 값이 서로 다름!! ==> 이로 인해 eq_resid를 계산할때 1e-5으로 수렴 못했던 문제가 발생함.
        ### 또한 이러한 문제가 obj value 계산할때도 영향을 미쳤을 수도 있음!        
        # self.genbase = ppc['gen'][:, idx_gen.MBASE]
        self.genbase = ppc['baseMVA']
        self.baseMVA = ppc['baseMVA']

        self.slack = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0]
        self.pv = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 2)[0] ## <== 버스안에 여러 발전기들 있는 case 존재.. fix 하기.
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
        flow_max[flow_max == 0] = np.inf # np.Inf
        self.line_limit = torch.tensor(flow_max, dtype=torch.get_default_dtype())

        ppc2 = deepcopy(ppc)
        Ybus, _, _ = makeYbus(self.baseMVA, ppc2['bus'], ppc2['branch'])
        Ybus = Ybus.todense()
        self.Ybusr = torch.tensor(np.real(Ybus), dtype=torch.get_default_dtype())
        self.Ybusi = torch.tensor(np.imag(Ybus), dtype=torch.get_default_dtype())
        
        ## Define optimization problem input and output variables
        demand = data['Dem'].T / self.baseMVA
        gen =  data['Gen'].T / self.genbase
        voltage = data['Vol'].T

        ## Graph data representation
        node_feat_x = np.zeros((demand.shape[0], self.nbus, 2))
        node_mask_x = np.zeros((demand.shape[0], self.nbus))
        for s in range(node_feat_x.shape[0]):
            node_feat_x[s,:,0] = np.real(demand[s,:])
            node_feat_x[s,:,1] = np.imag(demand[s,:])
            node_mask_x[s,self.spv] = 1

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
                edge_attr=edge_feat_x[i, :, 2:],)]
        
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


        ## Define train/valid/test split
        self._train_frac_semi = train_frac_semi
        self._valid_frac = valid_frac
        self._test_frac = test_frac


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

        # trace
        self.Sfmn_tr_list = None
        self.Sfmnbar_tr_list = None
        self.Stmn_tr_list = None
        self.Stmnbar_tr_list = None


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
        return self._G_data[:100]

    @property
    def valid_dataset(self):
        return self._G_data[100:1000]

    @property
    def test_dataset(self):
        return self._G_data[-200:]

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

    # def ineq_dist_back_log(self, X, Y): # using only for backpropagation
    #     _, _, vm, va = self.get_yvars(Y)
    #     cosva = torch.cos(va) # (batch, nbus)
    #     sinva = torch.sin(va) # (batch, nbus)
    #     vr = vm*cosva # (batch, nbus)
    #     vi = vm*sinva # (batch, nbus)
    #     v = torch.cat([vr, vi], dim = -1) # (batch, 2*nbus)
    #     #v = v_old.unsqueeze(0).expand(self.nl, *v_old.shape).transpose(0,1) # (batch, nl, 2*bus) # 이렇게 해도 되나...?

    #     #self.Sfmn_tr_list = Sf_tr
    #     #self.Sfmnbar_tr_list = Sfbar_tr
    #     #self.Stmn_tr_list = St_tr
    #     #self.Stmnbar_tr_list = Stbar_tr

    #     result = self.ineq_resid(X, Y)
    #     return result

    # def eq_grad(self, X, Y):
    #     eq_jac = self.eq_jac(Y)
    #     eq_resid = self.eq_resid(X,Y)
    #     return 2*eq_jac.transpose(1,2).bmm(eq_resid.unsqueeze(-1)).squeeze(-1)

    # def ineq_grad(self, X, Y, box_const_jac_dense):
    #     _, _, vm, va = self.get_yvars(Y)

    #     cosva = torch.cos(va) # (batch, nbus)
    #     sinva = torch.sin(va) # (batch, nbus)
    #     vr = vm*cosva # (batch, nbus)
    #     vi = vm*sinva # (batch, nbus)
    #     v = torch.cat([vr, vi], dim = -1) # (batch, 2*nbus)

    #     #v = v_old.unsqueeze(0).expand(self.nl, *v_old.shape).transpose(0,1) # (batch, nl, 2*bus) # 이렇게 해도 되나...?

    #     #self.Sfmn_tr_list = Sf_tr
    #     #self.Sfmnbar_tr_list = Sfbar_tr
    #     #self.Stmn_tr_list = St_tr
    #     #self.Stmnbar_tr_list = Stbar_tr

    #     ineq_jac = self.ineq_jac(Y, cosva, sinva, vr, vi, box_const_jac_dense)
    #     ineq_dist = self.ineq_dist(X, Y).to(dtype=torch.get_default_dtype())

    #     return 2*ineq_jac.transpose(1,2).bmm(ineq_dist.unsqueeze(-1)).squeeze(-1)


    # def ineq_partial_grad(self, X, Y, box_const_jac_dense):
    #     eq_jac = self.eq_jac(Y)
    #     dynz_dz = -torch.inverse(eq_jac[:, :, self.other_vars]).bmm(eq_jac[:, :, self.partial_vars])
    #     direct_grad = self.ineq_grad(X, Y, box_const_jac_dense)
    #     indirect_partial_grad = dynz_dz.transpose(1,2).bmm(
    #         direct_grad[:, self.other_vars].unsqueeze(-1)).squeeze(-1)

    #     full_partial_grad = indirect_partial_grad + direct_grad[:, self.partial_vars]

    #     full_grad = torch.zeros(X.shape[0], self.ydim, device=self.device)
    #     full_grad[:, self.partial_vars] = full_partial_grad
    #     full_grad[:, self.other_vars] = dynz_dz.bmm(full_partial_grad.unsqueeze(-1)).squeeze(-1)

    #     return full_grad

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
            
            # d_int = newton_jac_inv.transpose(1,2).bmm(
            #                 dl_dy_total[:,newton_guess_inds].unsqueeze(-1)).squeeze(-1)
            d_int = torch.linalg.solve(newton_jac_inv.transpose(1,2), dl_dy_total[:,newton_guess_inds].unsqueeze(-1)).squeeze(-1)

#             A_jac = cp.asarray(newton_jac_inv.transpose(1,2).detach().cpu().numpy())
#             b_jac = cp.asarray(dl_dy_total[:,newton_guess_inds].unsqueeze(-1).detach().cpu().numpy())
#             d_int = torch.tensor(cupy.linalg.solve(A_jac, b_jac).squeeze(-1), device=DEVICE)

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