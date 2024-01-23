import torch
import cvxpy as cp
import clarabel

def semi_UOT_KL(a,b,C,lam2,solver="ECOS"):
        
    n,m=C.shape  
    pi = cp.Variable((n,m))

    objective = cp.Minimize(cp.sum(cp.multiply(pi,C))
                            +lam2*cp.sum(cp.kl_div((pi.T)@torch.ones(n),b)))
                            
    constraints = [pi>=0,pi@torch.ones(m)==torch.ones(n)]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=solver)
    return pi.value

def semi_UOT_MMD(a,b,C,lam2,Ky=None,solver="ECOS",regul=1e-5):

    n,m=C.shape
    pi = cp.Variable((n,m))

    ### penalization MMD ###
    Ky+=regul*torch.eye(Ky.shape[0])
    objective = cp.Minimize(cp.sum(cp.multiply(pi,C))
                            +lam2*(cp.quad_form((pi.T)@torch.ones(n), Ky)-2*(((pi.T)@torch.ones(n))@Ky@b)))
    constraints = [pi>=0,pi@torch.ones(m)==torch.ones(n)]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=solver)
    return pi.value
