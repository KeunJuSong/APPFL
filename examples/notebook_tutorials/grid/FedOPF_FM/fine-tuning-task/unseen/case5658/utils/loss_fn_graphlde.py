import torch
####################### LD framework #######################
def total_loss(data, X, Y, LagM_slack_p_gen, LagM_gen, LagM_bus, LagM_line):
    X_ = torch.zeros((Y.shape[0], data.nbus*2)).to(data.device)
    for idx in range(X_.shape[0]):
        X_[idx,:data.nbus] = X[data.nbus*idx:data.nbus*(idx+1),0] # Pd
        X_[idx,data.nbus:] = X[data.nbus*idx:data.nbus*(idx+1),1] # Qd

    # data.x의 크기를 맞추어 reshape한 뒤, X에 바로 할당합니다.
    # reshaped_data = X.view(-1, data.nbus, 2)  # [num_samples, nbus, 2]
    # # 필요한 부분만 슬라이싱하여 X에 할당
    # X_[:, :data.nbus] = reshaped_data[:, :, 0]  # Pd
    # X_[:, data.nbus:] = reshaped_data[:, :, 1]  # Qd

    obj_cost = data.obj_fn(Y)

    ineq_dist = data.ineq_dist(X_, Y)
    # ineq_dist = data.ineq_dist_back(X, Y)
    # ineq_cost = LagM*ineq_dist # (batch, # of constraints)

    eq_resid = data.eq_resid(X_, Y)
    # eq_cost = eq_LagM*eq_resid # (batch, # of constraints)
    
    # ineq_cost = LagM*ineq_dist
    ineq_spg = ineq_dist[:,:2]
    ineq_qg = ineq_dist[:,2:2+2*data.ng]
    ineq_v_m = ineq_dist[:,2+2*data.ng:2+2*data.ng+2*data.nbus]
    ineq_line_l = ineq_dist[:,2+2*data.ng+2*data.nbus:]

    ineq_cost_spg = LagM_slack_p_gen*ineq_spg
    ineq_cost_qg = LagM_gen*ineq_qg
    ineq_cost_v_m = LagM_bus*ineq_v_m
    ineq_cost_line_l = LagM_line*ineq_line_l
    ineq_cost = torch.cat([ineq_cost_spg, ineq_cost_qg, ineq_cost_v_m, ineq_cost_line_l], dim=1)

    # ineq_cost_spg_sq_l2 = torch.square(ineq_spg)
    # ineq_cost_qg_sq_l2 = torch.square(ineq_qg)
    # ineq_cost_v_m_sq_l2 = torch.square(ineq_v_m)
    # ineq_cost_line_l_sq_l2 = torch.square(ineq_line_l)
    # ineq_cost_sq_l2 = torch.cat([ineq_cost_spg_sq_l2, ineq_cost_qg_sq_l2, ineq_cost_v_m_sq_l2, ineq_cost_line_l_sq_l2], dim=1)

    return obj_cost + ineq_cost.sum(dim = 1), obj_cost, ineq_dist, eq_resid
    # return 10*obj_cost + ineq_cost.sum(dim = 1), obj_cost, ineq_dist, eq_resid
    # return obj_cost + ineq_cost.sum(dim = 1) + ineq_cost_sq_l2.sum(dim=1), obj_cost, ineq_dist, eq_resid

####################### CONFIG method #######################
### PGLib 4601 Bus ###
# def total_loss(data, X, Y, LagM_1, LagM_2, LagM_3):
#     X_ = torch.zeros((Y.shape[0], data.nbus*2)).to(data.device)
#     for idx in range(X_.shape[0]):
#         X_[idx,:data.nbus] = X[data.nbus*idx:data.nbus*(idx+1),0] # Pd
#         X_[idx,data.nbus:] = X[data.nbus*idx:data.nbus*(idx+1),1] # Qd

#     obj_cost = data.obj_fn(Y)

#     ineq_dist = data.ineq_dist(X_, Y)
#     ineq_q_g = ineq_dist[:,:2*data.ng]
#     ineq_v_m = ineq_dist[:,2*data.ng:2*data.ng+2*data.nbus]
#     ineq_line_l = ineq_dist[:,2*data.ng+2*data.nbus:] 

#     eq_resid = data.eq_resid(X_, Y)

#     ineq_q_g_cost = LagM_1*ineq_q_g
#     ineq_v_m_cost = LagM_2*ineq_v_m
#     ineq_line_l_cost = LagM_3*ineq_line_l
    
#     return [obj_cost, ineq_q_g_cost.sum(dim = 1), ineq_v_m_cost.sum(dim = 1), ineq_line_l_cost.sum(dim = 1)], obj_cost, ineq_dist, eq_resid

### KPX 4872 Bus (DEPRECATED..!) ###
# def total_loss(data, X, Y, LagM_1, LagM_2):
#     X_ = torch.zeros((Y.shape[0], data.nbus*2)).to(data.device)
#     for idx in range(X_.shape[0]):
#         X_[idx,:data.nbus] = X[data.nbus*idx:data.nbus*(idx+1),0] # Pd
#         X_[idx,data.nbus:] = X[data.nbus*idx:data.nbus*(idx+1),1] # Qd

#     obj_cost = data.obj_fn(Y)

#     ineq_dist = data.ineq_dist(X_, Y)
#     ineq_q_g = ineq_dist[:,:2*data.ng]
#     ineq_v_m = ineq_dist[:,2*data.ng:2*data.ng+2*data.nbus]

#     eq_resid = data.eq_resid(X_, Y)

#     ineq_q_g_cost = LagM_1*ineq_q_g
#     ineq_v_m_cost = LagM_2*ineq_v_m
    
#     return [obj_cost, ineq_q_g_cost.sum(dim = 1), ineq_v_m_cost.sum(dim = 1)], obj_cost, ineq_dist, eq_resid

###########################################################################################################################################

####################### LD framework #######################
def ineq_violation(data, X, Y):
    X_ = torch.zeros((Y.shape[0], data.nbus*2)).to(data.device)
    for idx in range(X_.shape[0]):
        X_[idx,:data.nbus] = X[data.nbus*idx:data.nbus*(idx+1),0] # Pd
        X_[idx,data.nbus:] = X[data.nbus*idx:data.nbus*(idx+1),1] # Qd

    # # data.x의 크기를 맞추어 reshape한 뒤, X에 바로 할당합니다.
    # reshaped_data = X.view(-1, data.nbus, 2)  # [num_samples, nbus, 2]
    # # 필요한 부분만 슬라이싱하여 X에 할당
    # X_[:, :data.nbus] = reshaped_data[:, :, 0]  # Pd
    # X_[:, data.nbus:] = reshaped_data[:, :, 1]  # Qd

    ineq_dist = data.ineq_dist(X_, Y) # (batch, # of constraints)
    ineq_cost = ineq_dist.sum(dim = 0) # (1, # of constraints)
    return ineq_cost

####################### CONFIG method #######################
### PGLib 4601 Bus ###
# def ineq_violation(data, X, Y):
#     X_ = torch.zeros((Y.shape[0], data.nbus*2)).to(data.device)
#     for idx in range(X_.shape[0]):
#         X_[idx,:data.nbus] = X[data.nbus*idx:data.nbus*(idx+1),0] # Pd
#         X_[idx,data.nbus:] = X[data.nbus*idx:data.nbus*(idx+1),1] # Qd

#     # ineq_dist = data.ineq_dist_back(X, Y) # (batch, # of constraints)
#     ineq_dist = data.ineq_dist(X_, Y) # (batch, # of constraints)
#     ineq_q_g = ineq_dist[:,:2*data.ng]
#     ineq_v_m = ineq_dist[:,2*data.ng:2*data.ng+2*data.nbus]
#     ineq_line_l = ineq_dist[:,2*data.ng+2*data.nbus:]

#     # ineq_cost = ineq_dist.sum(dim = 0) # (1, # of constraints)
#     ineq_q_g_cost = ineq_q_g.sum(dim = 0) # (1, # of constraints)
#     ineq_v_m_cost = ineq_v_m.sum(dim = 0) # (1, # of constraints)
#     ineq_line_l_cost = ineq_line_l.sum(dim = 0) # (1, # of constraints)

#     # return ineq_cost
#     return ineq_q_g_cost, ineq_v_m_cost, ineq_line_l_cost

### KPX 4872 Bus (DEPRECATED..!) ###
# def ineq_violation(data, X, Y):
#     X_ = torch.zeros((Y.shape[0], data.nbus*2)).to(data.device)
#     for idx in range(X_.shape[0]):
#         X_[idx,:data.nbus] = X[data.nbus*idx:data.nbus*(idx+1),0] # Pd
#         X_[idx,data.nbus:] = X[data.nbus*idx:data.nbus*(idx+1),1] # Qd

#     # ineq_dist = data.ineq_dist_back(X, Y) # (batch, # of constraints)
#     ineq_dist = data.ineq_dist(X_, Y) # (batch, # of constraints)
#     ineq_q_g = ineq_dist[:,:2*data.ng]
#     ineq_v_m = ineq_dist[:,2*data.ng:2*data.ng+2*data.nbus]

#     # ineq_cost = ineq_dist.sum(dim = 0) # (1, # of constraints)
#     ineq_q_g_cost = ineq_q_g.sum(dim = 0) # (1, # of constraints)
#     ineq_v_m_cost = ineq_v_m.sum(dim = 0) # (1, # of constraints)

#     # return ineq_cost
#     return ineq_q_g_cost, ineq_v_m_cost
