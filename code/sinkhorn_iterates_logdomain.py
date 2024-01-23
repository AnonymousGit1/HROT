
import torch
import ot
from scipy.special import lambertw

def logsinkhorn(a,b,C,lam=50,eps=1,numiter=500,lam2=None,pen=None,
             Cx=None,Cy=None,lam3=None,eps2=None,numiter2=1,innerplan=False,Kx=None,Ky=None):
    loga=torch.log(a)
    logb=torch.log(b)
    f=loga
    M=-C/eps
    fy=torch.logsumexp(M + f[:,None],dim=0)
    fx=None
    if lam2 is None:
        lam2=lam
    if lam3 is None:
        lam3=lam
    if eps2 is None:
        eps2=eps
        
    if pen =="sinkhorn" or pen=="UOT_sinkhorn":
        Mx=-Cx/eps2
        My=-Cy/eps2
    if pen == "kkl":
	    Kxinv=torch.linalg.inv(lam*Kx+eps*torch.eye(Kx.shape[0]))
	    Kyinv=torch.linalg.inv(lam2*Ky+eps*torch.eye(Ky.shape[0]))
	    Kxloga=Kx@torch.log(a)
	    Kylogb=Ky@torch.log(b)
    
    for i in range(numiter):
        if pen is None:
            logprox=logb
        if pen=="kl":
            logprox=logprox_KL(torch.logsumexp(M + f[:,None],dim=0),logb,lam2,eps)
        if pen=="kkl":
            logprox=logprox_KKL(torch.logsumexp(M + f[:,None],dim=0),Kylogb,lam2,eps,Kyinv)
       #if pen=="l2":
       #     logprox=logprox_l2(K.T@u,b,lam2)          
        if pen=="sinkhorn":
            fy,gy=logprox_sinkhorn(torch.logsumexp(M + f[:,None],dim=0),logb,My,lam2/eps,eps2,numiter2,fy)
            Qy=torch.exp(fy.reshape((-1, 1)) + My + gy.reshape((1, -1)))
            logprox=torch.log(torch.sum(Qy,axis=1))
        #if pen=="UOT_sinkhorn":
        #    fy,gy=logprox_UOT_sinkhorn(torch.logsumexp(M + f[:,None],dim=0),logb,My,lam2/eps,eps2,numiter2,lam3,fy)
        #    Qy=torch.exp(fy.reshape((-1, 1)) + My + gy.reshape((1, -1)))
        #    logprox=torch.log(torch.sum(Qy,axis=1))
                                       
        g = logprox - torch.logsumexp(M + f[:,None],dim=0)
        
        
        if pen is None:
            logprox=loga
        if pen=="kl":
            logprox=logprox_KL(torch.logsumexp(M + g[None,:],dim=1),loga,lam,eps)
        if pen=="kkl":
            logprox=logprox_KKL(torch.logsumexp(M + g[None,:],dim=1),Kxloga,lam,eps,Kxinv)
        #if pen=="l2":
        #    logprox=logprox_l2(K@v,a,lam)           
        if pen=="sinkhorn":
            fx,gx=logprox_sinkhorn(torch.logsumexp(M + g[None,:],dim=1),loga,Mx,lam/eps,eps2,numiter2,fx)
            Qx=torch.exp(fx.reshape((-1, 1)) + Mx + gx.reshape((1, -1)))
            logprox=torch.log(torch.sum(Qx,axis=1))
        #if pen=="UOT_sinkhorn":
        #    fx,gx,Mx=logprox_UOT_sinkhorn(torch.logsumexp(M + g[None,:],dim=1),loga,Cx,lam/eps,eps2,numiter2,lam3,fx)
        #    Qx=torch.exp(fx.reshape((-1, 1)) + Mx + gx.reshape((1, -1)))
        #    logprox=torch.log(torch.sum(Qx,axis=1))
  
        f = logprox - torch.logsumexp(M + g[None,:],dim=1)
    if innerplan:
        return torch.exp(f),torch.exp(g),torch.exp(M),Qx,Qy
    else:
        return torch.exp(f),torch.exp(g),torch.exp(M)
    

def logprox_KL(Mf,logb,lam,eps):
    gam1=lam/(eps+lam)
    gam2=eps/(eps+lam)
    return gam2*Mf+gam1*logb
    
def logprox_KKL(Mf,Klog,lam,eps,Kinv):
    return Kinv@(eps*Mf+lam*Klog)  
    
def logprox_l2(Kv,b,lam):
    return (torch.real(lambertw(lam*Kv*torch.exp(lam*b), k=0, tol=1e-8))/lam).float()
  
        
def logprox_sinkhorn(loga,logb,M,lam,eps2,numiter,f):
    if f is None:
        f=loga
    gamma=lam/(eps2+lam)
    for i in range(numiter):
        #g = logb - torch.logsumexp(M + f[:,None],dim=0)
        #f = gamma*(loga - torch.logsumexp(M + g[None,:],dim=1))
        g = logb - torch.logsumexp(M + f[:,None],dim=0)
        f = gamma*(loga - torch.logsumexp(M + g[None,:],dim=1))
    g = logb - torch.logsumexp(M + f[:,None],dim=0)
    return f,g
 
    
def logprox_UOT_sinkhorn(loga,logb,M,lam1,eps2,numiter,lam3,f):
    if f is None: 
        f=loga
    gamma1=lam1/(eps2+lam1)
    gamma2=lam3/(eps2+lam3)
    #print(torch.sum(Kv),torch.sum(b))
    for i in range(numiter):
        g = gamma2*(logb - torch.logsumexp(M + f[:,None],dim=0))
        f = gamma1*(loga - torch.logsumexp(M + g[None,:],dim=1))
    g = gamma2*(logb - torch.logsumexp(M + f[:,None],dim=0))
    return torch.exp(f.reshape((-1, 1)) + M + g.reshape((1, -1)))




"""
import torch
import ot
from scipy.special import lambertw

def logsinkhorn(a,b,C,lam=50,eps=1,numiter=500,lam2=None,pen=None,
             Cx=None,Cy=None,lam3=None,eps2=None,numiter2=1,innerplan=False):
    loga=torch.log(a)
    logb=torch.log(b)
    f=loga
    M=-C/eps
    fy=torch.logsumexp(M + f[:,None],dim=0)
    fx=None
    if lam2 is None:
        lam2=lam
    if lam3 is None:
        lam3=lam
    if eps2 is None:
        eps2=eps
    
    for i in range(numiter):
        if pen is None:
            logprox=logb
        if pen=="kl":
            logprox=logprox_KL(torch.logsumexp(M + f[:,None],dim=0),logb,lam2,eps)
       #if pen=="l2":
       #     logprox=logprox_l2(K.T@u,b,lam2)          
        if pen=="sinkhorn":
            fy,gy,My=logprox_sinkhorn(torch.logsumexp(M + f[:,None],dim=0),logb,Cy,lam2/eps,eps2,numiter2,fy)
            Qy=torch.exp(fy.reshape((-1, 1)) + My + gy.reshape((1, -1)))
            logprox=torch.log(torch.sum(Qy,axis=1))
        #if pen=="UOT_sinkhorn":
        #    fy,gy,My=logprox_UOT_sinkhorn(torch.logsumexp(M + f[:,None],dim=0),logb,Cy,lam2/eps,eps2,numiter2,lam3,fy)
        #    Qy=torch.exp(fy.reshape((-1, 1)) + My + gy.reshape((1, -1)))
        #    logprox=torch.log(torch.sum(Qy,axis=1))
                                       
        g = logprox - torch.logsumexp(M + f[:,None],dim=0)
        
        
        if pen is None:
            logprox=loga
        if pen=="kl":
            logprox=logprox_KL(torch.logsumexp(M + g[None,:],dim=1),loga,lam,eps)
        #if pen=="l2":
        #    logprox=logprox_l2(K@v,a,lam)           
        if pen=="sinkhorn":
            fx,gx,Mx=logprox_sinkhorn(torch.logsumexp(M + g[None,:],dim=1),loga,Cx,lam/eps,eps2,numiter2,fx)
            Qx=torch.exp(fx.reshape((-1, 1)) + Mx + gx.reshape((1, -1)))
            logprox=torch.log(torch.sum(Qx,axis=1))
        #if pen=="UOT_sinkhorn":
        #    fx,gx,Mx=logprox_UOT_sinkhorn(torch.logsumexp(M + g[None,:],dim=1),loga,Cx,lam/eps,eps2,numiter2,lam3,fx)
        #    Qx=torch.exp(fx.reshape((-1, 1)) + Mx + gx.reshape((1, -1)))
        #    logprox=torch.log(torch.sum(Qx,axis=1))
  
        f = logprox - torch.logsumexp(M + g[None,:],dim=1)
    if innerplan:
        return torch.exp(f),torch.exp(g),torch.exp(M),Qx,Qy
    else:
        return torch.exp(f),torch.exp(g),torch.exp(M)
    

def logprox_KL(Mf,logb,lam,eps):
    gam1=lam/(eps+lam)
    gam2=eps/(eps+lam)
    return gam2*Mf+gam1*logb
    
def logprox_l2(Kv,b,lam):
    return (torch.real(lambertw(lam*Kv*torch.exp(lam*b), k=0, tol=1e-8))/lam).float()
  
        
def logprox_sinkhorn(loga,logb,C,lam,eps2,numiter,f):
    if f is None:
        f=loga
    M=-C/eps2 #M symmetric matrix since C is
    gamma=lam/(eps2+lam)
    for i in range(numiter):
        #g = logb - torch.logsumexp(M + f[:,None],dim=0)
        #f = gamma*(loga - torch.logsumexp(M + g[None,:],dim=1))
        g = logb - torch.logsumexp(M + f[:,None],dim=0)
        f = gamma*(loga - torch.logsumexp(M + g[None,:],dim=1))
    g = logb - torch.logsumexp(M + f[:,None],dim=0)
    return f,g,M
 
    
def logprox_UOT_sinkhorn(loga,logb,C,lam1,eps2,numiter,lam3,f):
    if f is None: 
        f=loga
    M=-C/eps2#K symmetric matrix since C is
    gamma1=lam1/(eps2+lam1)
    gamma2=lam3/(eps2+lam3)
    #print(torch.sum(Kv),torch.sum(b))
    for i in range(numiter):
        g = gamma2*(logb - torch.logsumexp(M + f[:,None],dim=0))
        f = gamma1*(loga - torch.logsumexp(M + g[None,:],dim=1))
    g = gamma2*(logb - torch.logsumexp(M + f[:,None],dim=0))
    return torch.exp(f.reshape((-1, 1)) + M + g.reshape((1, -1)))
"""
