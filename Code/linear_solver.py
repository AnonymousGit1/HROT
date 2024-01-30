import torch
import cvxpy as cp
import ot
import clarabel

    
    
def UOT_W(a,b,C,lam,lam2=None,Cx=None,Cy=None,innerplan=False,solver="ECOS"):
    if lam2 is None:
        lam2=lam

    n,m=C.shape
    
        
    pi = cp.Variable((n,m))
    Qx=cp.Variable((n,n))
    Qy=cp.Variable((m,m))

    ### penalization Wasserstein ###
    objective = cp.Minimize(cp.sum(cp.multiply(pi,C))
                +lam*cp.sum(cp.multiply(Qx,Cx))
                +lam2*cp.sum(cp.multiply(Qy,Cy)))
    constraints = [pi>=0,Qx>=0,Qy>=0,
                   Qx@torch.ones(n)==pi@torch.ones(m),
                   Qy@torch.ones(m)==(pi.T)@torch.ones(n),
                   Qx.T@torch.ones(n)==a,
                   Qy.T@torch.ones(m)==b]

    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=solver)
    if innerplan:
        return pi.value,Qx.value,Qy.value
    else:
        return pi.value
   
def UOT_KL(a,b,C,lam,lam2=None,solver="ECOS"):
    if lam2 is None:
        lam2=lam
        
    n,m=C.shape  
    pi = cp.Variable((n,m))

    objective = cp.Minimize(cp.sum(cp.multiply(pi,C))
                            +lam*cp.sum(cp.kl_div(pi@torch.ones(m),a))
                            +lam2*cp.sum(cp.kl_div((pi.T)@torch.ones(n),b)))
                            
    constraints = [pi>=0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=solver)
    return pi.value

