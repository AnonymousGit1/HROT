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
        
def UOT_MMD(a,b,C,lam,lam2=None,Kx=None,Ky=None,solver="ECOS",regul=1e-5):
    if lam2 is None:
        lam2=lam

    n,m=C.shape
    pi = cp.Variable((n,m))

    ### penalization MMD ###
    Kx+=regul*torch.eye(Kx.shape[0])
    Ky+=regul*torch.eye(Ky.shape[0])
    objective = cp.Minimize(cp.sum(cp.multiply(pi,C))
                            +lam*(cp.quad_form(pi@torch.ones(m), Kx)-2*((pi@torch.ones(m))@Kx@a))
                            +lam2*(cp.quad_form((pi.T)@torch.ones(n), Ky)-2*(((pi.T)@torch.ones(n))@Ky@b)))
    constraints = [pi>=0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=solver)
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



"""
def UOT_Wsq_C(X,Y,lam,a=None,b=None,lam2=None):
    if lam2 is None:
        lam2=lam

    C=ot.dist(X,Y)
    Cs=ot.dist(X,X,metric='euclidean')
    Ct=ot.dist(Y,Y,metric='euclidean')
    #Cs=ot.dist(X,X,metric='minkowski',p=1)
    #Ct=ot.dist(Y,Y,metric='minkowski',p=1)
    n,m=C.shape
    
    if a is None:
        a=torch.ones((n,))/n
    if b is None:
        b=torch.ones((m,))/m
        
    pi = cp.Variable((n,m))
    pis=cp.Variable((n,n))
    pit=cp.Variable((m,m))

    ### penalization Wasserstein ###
    objective = cp.Minimize(cp.sum(cp.multiply(pi,C))
                +lam*cp.sum(cp.multiply(pis,Cs))
                +lam2*cp.sum(cp.multiply(pit,Ct)))
    constraints = [pi>=0,pis>=0,pit>=0,
                   pis@torch.ones(n)==pi@torch.ones(m),
                   pit@torch.ones(m)==(pi.T)@torch.ones(n),
                   pis.T@torch.ones(n)==a,
                   pit.T@torch.ones(m)==b]

    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return prob.value,pi.value,pis.value,pit.value
    
def UOT_W_Csq(X,Y,lam,a=None,b=None,lam2=None):
    if lam2 is None:
        lam2=lam

    C=ot.dist(X,Y)
    Cs=ot.dist(X,X)
    Ct=ot.dist(Y,Y)
    n,m=C.shape
    
    if a is None:
        a=torch.ones((n,))/n
    if b is None:
        b=torch.ones((m,))/m
        
    pi = cp.Variable((n,m))
    pis=cp.Variable((n,n))
    pit=cp.Variable((m,m))

    ### penalization Wasserstein ###
    objective = cp.Minimize(cp.sum(cp.multiply(pi,C))
                +lam*cp.power(cp.sum(cp.multiply(pis,Cs)),.5)
                +lam2*cp.power(cp.sum(cp.multiply(pit,Ct)),.5))
    constraints = [pi>=0,pis>=0,pit>=0,
                   pis@torch.ones(n)==pi@torch.ones(m),
                   pit@torch.ones(m)==(pi.T)@torch.ones(n),
                   pis.T@torch.ones(n)==a,
                   pit.T@torch.ones(m)==b]

    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return prob.value,pi.value,pis.value,pit.value

def UOT_W_C(X,Y,lam,a=None,b=None,lam2=None):
    if lam2 is None:
        lam2=lam

    C=ot.dist(X,Y)
    Cs=ot.dist(X,X,metric='euclidean')
    Ct=ot.dist(Y,Y,metric='euclidean')
    n,m=C.shape
    
    if a is None:
        a=torch.ones((n,))/n
    if b is None:
        b=torch.ones((m,))/m
        
    pi = cp.Variable((n,m))
    pis=cp.Variable((n,n))
    pit=cp.Variable((m,m))

    ### penalization Wasserstein ###
    objective = cp.Minimize(cp.sum(cp.multiply(pi,C))
                +lam*cp.power(cp.sum(cp.multiply(pis,Cs)),.5)
                +lam2*cp.power(cp.sum(cp.multiply(pit,Ct)),.5))
    constraints = [pi>=0,pis>=0,pit>=0,
                   pis@torch.ones(n)==pi@torch.ones(m),
                   pit@torch.ones(m)==(pi.T)@torch.ones(n),
                   pis.T@torch.ones(n)==a,
                   pit.T@torch.ones(m)==b]

    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return prob.value,pi.value,pis.value,pit.value
    
def UOT_Wsq_Csq_l2sq(X,Y,lam,a=None,b=None,lam2=None):
    if lam2 is None:
        lam2=lam

    C=ot.dist(X,Y)
    Cs=ot.dist(X,X)
    Ct=ot.dist(Y,Y)
    n,m=C.shape
    
    if a is None:
        a=torch.ones((n,))/n
    if b is None:
        b=torch.ones((m,))/m
        
    pi = cp.Variable((n,m))
    pit=cp.Variable((m,m))

    ### penalization Wasserstein ###
    objective = cp.Minimize(cp.sum(cp.multiply(pi,C))
                +lam*cp.norm(pi@torch.ones(m)-a,2)
                +lam2*cp.sum(cp.multiply(pit,Ct)))
    constraints = [pi>=0,pit>=0,
                   pit@torch.ones(m)==(pi.T)@torch.ones(n),
                   pit.T@torch.ones(m)==b]

    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return prob.value,pi.value,pit.value
        
def UOT_KL(X,Y,lam,a=None,b=None,lam2=None):
    if lam2 is None:
        lam2=lam
        
    C=ot.dist(X,Y)
    n,m=C.shape
    
    if a is None:
        a=torch.ones((n,))/n
    if b is None:
        b=torch.ones((m,))/m
        
    pi = cp.Variable((n,m))

    objective = cp.Minimize(cp.sum(cp.multiply(pi,C))
                            +lam*cp.sum(cp.kl_div(pi@torch.ones(m),a))
                            +lam2*cp.sum(cp.kl_div((pi.T)@torch.ones(n),b)))
                            
    constraints = [pi>=0]

    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return prob.value,pi.value


    
def UOT_l2(X,Y,lam,a=None,b=None,lam2=None):
    if lam2 is None:
        lam2=lam
    C=ot.dist(X,Y)
    n,m=C.shape
    
    if a is None:
        a=torch.ones((n,))/n
    if b is None:
        b=torch.ones((m,))/m
    
    pi = cp.Variable((n,m))
    objective = cp.Minimize(cp.sum(cp.multiply(pi,C))
                            +lam*cp.norm(pi@torch.ones(m)-a,2)
                            +lam2*cp.norm((pi.T)@torch.ones(n)-b,2))
    constraints = [pi>=0]
        
    prob = cp.Problem(objective, constraints)
    result = prob.solve()          
    return prob.value,pi.value  
    
def R_l2(X,Y,lam,a=None,b=None,lam2=None):
    if lam2 is None:
        lam2=lam
        
    C=ot.dist(X,Y)
    n,m=C.shape
    
    if a is None:
        a=torch.ones((n,))/n
    if b is None:
        b=torch.ones((m,))/m
    
    pi = cp.Variable((n,m))
    objective = cp.Minimize(cp.sum(cp.multiply(pi,C))
                            +lam*cp.norm(pi@torch.ones(m)-a,2)
                            +lam2*cp.norm((pi.T)@torch.ones(n)-b,2))
    constraints = [pi>=0,(pi@torch.ones(m))@torch.ones(n)==1]
        
    prob = cp.Problem(objective, constraints)
    result = prob.solve()          
    return prob.value,pi.value  

def UOT_l2sq(X,Y,lam,a=None,b=None,lam2=None):
    if lam2 is None:
        lam2=lam
        
    C=ot.dist(X,Y)
    n,m=C.shape
    
    if a is None:
        a=torch.ones((n,))/n
    if b is None:
        b=torch.ones((m,))/m
    
    pi = cp.Variable((n,m))
    objective = cp.Minimize(cp.sum(cp.multiply(pi,C))
                            +lam*cp.sum_squares(pi@torch.ones(m)-a)
                            +lam2*cp.sum_squares((pi.T)@torch.ones(n)-b))
    constraints = [pi>=0]
        
    prob = cp.Problem(objective, constraints)
    result = prob.solve()          
    return prob.value,pi.value  
    
def R_l2sq(X,Y,lam,a=None,b=None,lam2=None):
    if lam2 is None:
        lam2=lam
        
    C=ot.dist(X,Y)
    n,m=C.shape
    
    if a is None:
        a=torch.ones((n,))/n
    if b is None:
        b=torch.ones((m,))/m
    
    pi = cp.Variable((n,m))
    objective = cp.Minimize(cp.sum(cp.multiply(pi,C))
                            +lam*cp.sum_squares(pi@torch.ones(m)-a)
                            +lam2*cp.sum_squares((pi.T)@torch.ones(n)-b))
    constraints = [pi>=0,(pi@torch.ones(m))@torch.ones(n)==1]
        
    prob = cp.Problem(objective, constraints)
    result = prob.solve()          
    return prob.value,pi.value """ 
