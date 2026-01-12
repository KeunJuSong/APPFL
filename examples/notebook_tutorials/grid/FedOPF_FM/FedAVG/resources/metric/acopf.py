import numpy as np
import torch
from pypower.api import makeYbus

def acopf_feasibility(data, y_pred, input):
    """
    y_true and y_pred are both of type np.ndarray <== in this case, the type gonna be torch.tensor
    1) Feasibility violation
    2) Feasibility satisfaction
    """
    data._device = y_pred.device # Sync the device
    X_ = torch.zeros((y_pred.shape[0], data.nbus*2)).to(data.device)
    for idx in range(X_.shape[0]):
        X_[idx,:data.nbus] = input[data.nbus*idx:data.nbus*(idx+1),0] # Pd
        X_[idx,data.nbus:] = input[data.nbus*idx:data.nbus*(idx+1),1] # Qd

    Ybus, Yf, Yt = makeYbus(data.baseMVA, data.ppc['bus'], data.ppc['branch'])
    Ybusr = torch.tensor(np.real(Ybus.todense()), dtype=torch.get_default_dtype(), device=data.device)
    Ybusi = torch.tensor(np.imag(Ybus.todense()), dtype=torch.get_default_dtype(), device=data.device)

    # branch thermal limit information
    flow_max = (data.ppc['branch'][:, 5] / data.baseMVA)**2
    flow_max[flow_max == 0] = np.inf
    flow_max = torch.tensor(flow_max, dtype=torch.float32).to(data.device)

    pg, qg, vm, va = data.get_yvars(y_pred)
    vr = vm*torch.cos(va)
    vi = vm*torch.sin(va)
    vz = torch.complex(vr, vi) # complex voltage

    # calculate the branch current of from bus and to bus based on the Yf*V and Yt*V
    If = torch.tensor(Yf.todense(), dtype=torch.complex64).to(data.device) @ vz.T
    It = torch.tensor(Yt.todense(), dtype=torch.complex64).to(data.device) @ vz.T

    # Calculate the apparent power S
    Sf = vz[:,data.ppc['branch'][:,0].astype(int)] * torch.conj(If.T)
    St = vz[:,data.ppc['branch'][:,1].astype(int)] * torch.conj(It.T)
    Sff = Sf * torch.conj(Sf)
    Stt = St * torch.conj(St)

    # calculate the line thermal limit constraints violation
    diff_Sf = Sff.real - flow_max
    diff_St = Stt.real - flow_max
    # diff_Sf[torch.clamp(diff_Sf, 0) != 0]
    
    ## Power flow equation
    tmp1 = vr@(Ybusr.to(device=data.device, dtype=torch.get_default_dtype())) - vi@(Ybusi.to(device=data.device, dtype=torch.get_default_dtype()))
    tmp2 = -vr@(Ybusi.to(device=data.device, dtype=torch.get_default_dtype())) - vi@(Ybusr.to(device=data.device, dtype=torch.get_default_dtype()))
    # real power
    pg_expand = torch.zeros(pg.shape[0], data.nbus, device=data.device)
    pg_expand[:, data.spv] = pg
    real_resid = (pg_expand - X_[:, :data.nbus]) - (vr*tmp1 - vi*tmp2)
    # reactive power
    qg_expand = torch.zeros(qg.shape[0], data.nbus, device=data.device)
    qg_expand[:, data.spv] = qg
    react_resid = (qg_expand - X_[:, data.nbus:]) - (vr*tmp2 + vi*tmp1)
    ## all residuals
    resids = torch.cat([
        real_resid,
        react_resid
    ], dim=1)

    # line_limit_vio_Sf = torch.clamp(diff_Sf, 0)
    # line_limit_vio_St = torch.clamp(diff_St, 0)

    ## Objective cost
    obj_cost = data.obj_fn(y_pred)

    ## Feasibility violation
    test_ineq_p_g = torch.cat([pg - data.pmax.to(device=data.device), data.pmin.to(device=data.device) - pg], dim=1)
    test_ineq_p_g = torch.clamp(test_ineq_p_g, 0).to(data.device)
    test_ineq_q_g = torch.cat([qg - data.qmax.to(device=data.device), data.qmin.to(device=data.device) - qg], dim=1)
    test_ineq_q_g = torch.clamp(test_ineq_q_g, 0).to(data.device)
    test_ineq_v_m = torch.cat([vm - data.vmax.to(device=data.device), data.vmin.to(device=data.device) - vm], dim=1)
    test_ineq_v_m = torch.clamp(test_ineq_v_m, 0).to(data.device)
    test_ineq_line_l = torch.cat([Sff.real - flow_max, Stt.real - flow_max], dim=1)
    test_ineq_line_l = torch.clamp(test_ineq_line_l, 0).to(data.device)

    test_ineq_p_g_max = torch.max(test_ineq_p_g, dim=1)[0].detach().cpu().numpy()
    test_ineq_p_g_mean = torch.mean(test_ineq_p_g, dim=1).detach().cpu().numpy()
    test_ineq_q_g_max = torch.max(test_ineq_q_g, dim=1)[0].detach().cpu().numpy()
    test_ineq_q_g_mean = torch.mean(test_ineq_q_g, dim=1).detach().cpu().numpy()
    test_ineq_v_m_max = torch.max(test_ineq_v_m, dim=1)[0].detach().cpu().numpy()
    test_ineq_v_m_mean = torch.mean(test_ineq_v_m, dim=1).detach().cpu().numpy()
    test_ineq_line_l_max = torch.max(test_ineq_line_l, dim=1)[0].detach().cpu().numpy()
    test_ineq_line_l_mean = torch.mean(test_ineq_line_l, dim=1).detach().cpu().numpy()

    test_eq_active_pf_max = torch.max(torch.abs(real_resid), dim=1)[0].detach().cpu().numpy()
    test_eq_reactive_pf_max = torch.max(torch.abs(react_resid), dim=1)[0].detach().cpu().numpy()
    test_eq_active_pf_mean = torch.mean(torch.abs(real_resid), dim=1).detach().cpu().numpy()
    test_eq_reactive_pf_mean = torch.mean(torch.abs(react_resid), dim=1).detach().cpu().numpy()
        
    feas_vio_max = [test_ineq_p_g_max, test_ineq_q_g_max, test_ineq_v_m_max, test_ineq_line_l_max, test_eq_active_pf_max, test_eq_reactive_pf_max]
    feas_vio_mean = [test_ineq_p_g_mean, test_ineq_q_g_mean, test_ineq_v_m_mean, test_ineq_line_l_mean, test_eq_active_pf_mean, test_eq_reactive_pf_mean]

    ## Feasibility satisfaction
    pg_rate_torch = (torch.sum((pg <= data.pmax.to(device=data.device)) & (pg >= data.pmin.to(device=data.device)), dim=1)/data.ng*100).detach().cpu().numpy()
    qg_rate_torch = (torch.sum((qg <= data.qmax.to(device=data.device)) & (qg >= data.qmin.to(device=data.device)), dim=1)/data.ng*100).detach().cpu().numpy()
    v_rate_torch = (torch.sum((vm <= data.vmax.to(device=data.device)) & (vm >= data.vmin.to(device=data.device)), dim=1)/data.nbus*100).detach().cpu().numpy()
    sff_rate_torch = (torch.sum(Sff.real <= flow_max, dim=1)/data.nl*100).detach().cpu().numpy()
    stt_rate_torch = (torch.sum(Stt.real <= flow_max, dim=1)/data.nl*100).detach().cpu().numpy()

    test_p_g_satisfication = pg_rate_torch
    test_q_g_satisfication = qg_rate_torch
    test_v_m_satisfication = v_rate_torch
    test_l_sf_satisfication = sff_rate_torch
    test_l_st_satisfication = stt_rate_torch

    test_active_pf_satisfication = (torch.sum((real_resid <= 1e-2) & (real_resid >= -1e-2)  , dim=1)/real_resid.shape[1]*100).detach().cpu().numpy()
    test_reactive_pf_satisfication = (torch.sum((react_resid <= 1e-2) & (react_resid >= -1e-2)  , dim=1)/react_resid.shape[1]*100).detach().cpu().numpy()
    
    feas_sat = [test_p_g_satisfication, test_q_g_satisfication, test_v_m_satisfication, test_l_sf_satisfication, test_l_st_satisfication, test_active_pf_satisfication, test_reactive_pf_satisfication]

    return obj_cost, feas_vio_max, feas_vio_mean, feas_sat 