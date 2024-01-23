import torch
import ot
from scipy.special import lambertw
    

def sinkhorn(a,b,C,lam=50,eps=1,numiter=500,lam2=None,pen=None,
             Cx=None,Cy=None,lam3=None,eps2=None,numiter2=1,innerplan=False,Kx=None,Ky=None,lr=1e-3,thr=1e-5):
    u_old=a
    v_old=b
    u=a
    G=torch.exp(-C/eps)
    uy=G.T@u
    ux=None
    if lam2 is None:
        lam2=lam
    if lam3 is None:
        lam3=lam
    if eps2 is None:
        eps2=eps
    if pen =="sinkhorn" or pen=="UOT_sinkhorn":
        Gx=torch.exp(-Cx/eps2)
        Gy=torch.exp(-Cy/eps2)
        
    if pen == "kkl":
	    Kxinv=torch.linalg.inv(lam*Kx+eps*torch.eye(Kx.shape[0]))
	    Kyinv=torch.linalg.inv(lam2*Ky+eps*torch.eye(Ky.shape[0]))
	    Kxloga=Kx@torch.log(a)
	    Kylogb=Ky@torch.log(b)
    for i in range(numiter):
        if pen is None:
            prox=b
        elif pen=="kl":
            prox=prox_KL(G.T@u,b,lam2,eps)
        elif pen=="kkl":
            prox=prox_KKL(G.T@u,Kylogb,lam2,eps,Kyinv)
        elif pen=="l2":
            prox=prox_l2(G.T@u,b,lam2)
        elif pen=="MMD":
            prox=prox_MMD(G.T@u,b,Ky,lam2,eps,uy,numiter2,lr)
            uy=prox.clone()
        elif pen=="sinkhorn":
            uy,vy=prox_sinkhorn(G.T@u,b,Gy,lam2/eps,eps2,numiter2,uy)
            Qy=uy.reshape((-1, 1)) * Gy * vy.reshape((1, -1)) 
            prox=torch.sum(Qy,axis=1)
        elif pen=="UOT_sinkhorn":
            uy,vy=prox_UOT_sinkhorn(G.T@u,b,Gy,lam2/eps,eps2,numiter2,lam3,uy)
            Qy=uy.reshape((-1, 1)) * Gy * vy.reshape((1, -1))
            prox=torch.sum(Qy,axis=1)
                                   
        v=prox/(G.T@u)
        
        if pen is None:
            prox=a
        elif pen=="kl":
            prox=prox_KL(G@v,a,lam,eps)
        elif pen=="kkl":
            prox=prox_KKL(G@v,Kxloga,lam,eps,Kxinv)
        elif pen=="l2":
            prox=prox_l2(G@v,a,lam)
        if pen=="MMD":
            prox=prox_MMD(G@v,a,Kx,lam,eps,ux,numiter2,lr)
            ux=prox.clone()
        elif pen=="sinkhorn":
            ux,vx=prox_sinkhorn(G@v,a,Gx,lam/eps,eps2,numiter2,ux)
            Qx=ux.reshape((-1, 1)) * Gx * vx.reshape((1, -1)) 
            prox=torch.sum(Qx,axis=1)
        elif pen=="UOT_sinkhorn":
            ux,vx=prox_UOT_sinkhorn(G@v,a,Gx,lam/eps,eps2,numiter2,lam3,ux)
            Qx=ux.reshape((-1, 1)) * Gx * vx.reshape((1, -1))
            prox=torch.sum(Qx,axis=1) 
  
        u=prox/(G@v)
    
        if (torch.norm(u_old-u)+torch.norm(v-v_old))<thr:
                if innerplan:
                    return u,v,G,Qx,Qy
                else:
                    return u,v,G
        else:
            u_old=u
            v_old=v
    if innerplan:
        return u,v,G,Qx,Qy
    else:
        return u,v,G


      
def prox_KL(Gu,b,lam,eps):
    gam1=lam/(eps+lam)
    gam2=eps/(eps+lam)
    return (Gu**gam2)*(b**gam1)

def prox_KKL(Gu,Klogb,lam,eps,Kinv):
    return torch.exp(Kinv@(eps*torch.log(Gu)+lam*Klogb))    

def prox_l2(Gv,b,lam):
    return (torch.real(lambertw(lam*Gv*torch.exp(lam*b), k=0, tol=1e-8))/lam).float()

def prox_MMD(Gv,b,K,lam,eps,u_warm=None,numiter=100,lr=1e-3):
    if u_warm is None:# warmstart
        u_warm=Gv
    u,loss_l=logGD_MMDpen(Gv,b,K,lam,eps,u_warm,numiter,lr)
    #pl.plot(loss_l)
    return u
def MMDpen(u,Gv,b,K,lam,eps):
    return eps*(torch.sum(u*torch.log(u/Gv)-u+Gv))+lam*(b@K@b+u@K@u-2*b@K@u)


def loggrad_MMD(f,Gv,b,K,lam,eps):
    return eps*(f-torch.log(Gv))+2*lam*K@(torch.exp(f)-b)
  
def logGD_MMDpen(Gv,b,K,lam,eps,u_warm,numiter=100,lr=1e-2):
    f=torch.log(u_warm)
    loss_l=[]
    for i in range(numiter):
        f_next=f-lr*loggrad_MMD(f,Gv,b,K,lam,eps)
        f=f_next.clone()
        loss_l+=[MMDpen(torch.exp(f),Gv,b,K,lam,eps)]
    return torch.exp(f),loss_l
        
def prox_sinkhorn(Gv,b,G,lam,eps2,numiter,u): 
    if u is None:# warmstart
        u=Gv
    gamma=lam/(eps2+lam)
    u_old=u
    v_old=b
    for i in range(numiter):
        v=b/(G@u)
        u=(Gv/(G@v))**gamma
        if (torch.norm(u_old-u)+torch.norm(v-v_old))<1e-5:
            v=b/(G@u)
            return u,v
        else:
            u_old=u
            v_old=v
    v=b/(G@u)
    return u,v
    
def prox_UOT_sinkhorn(Gv,b,G,lam1,eps2,numiter,lam2,u):
    if u is None:
        u=Gv
    gamma1=lam1/(eps2+lam1)
    gamma2=lam2/(eps2+lam2)
    u_old=u
    v_old=b
    for i in range(numiter):
        v=(b/(G@u))**gamma2
        u=(Gv/(G@v))**gamma1
        if (torch.norm(u_old-u)+torch.norm(v-v_old))<1e-5:
            v=b/(G@u)
            return u,v
        else:
            u_old=u
            v_old=v
    return u,v
    



"""
import torch
import ot
from scipy.special import lambertw
    

def sinkhorn(a,b,C,lam=50,eps=1,numiter=500,lam2=None,pen=None,
             Cx=None,Cy=None,lam3=None,eps2=None,numiter2=1,innerplan=False,Kx=None,Ky=None,lr=1e-3,thr=1e-5):
    u_old=a
    v_old=b
    u=a
    G=torch.exp(-C/eps)
    uy=G.T@u
    ux=None
    if lam2 is None:
        lam2=lam
    if lam3 is None:
        lam3=lam
    if eps2 is None:
        eps2=eps
    for i in range(numiter):
        if pen is None:
            prox=b
        elif pen=="kl":
            prox=prox_KL(G.T@u,b,lam2,eps)
        elif pen=="l2":
            prox=prox_l2(G.T@u,b,lam2)
        elif pen=="MMD":
            prox=prox_MMD(G.T@u,b,Ky,lam2,eps,uy,numiter2,lr)
            uy=prox.clone()
        elif pen=="sinkhorn":
            uy,vy,Gy=prox_sinkhorn(G.T@u,b,Cy,lam2/eps,eps2,numiter2,uy)
            Qy=uy.reshape((-1, 1)) * Gy * vy.reshape((1, -1)) 
            prox=torch.sum(Qy,axis=1)
        elif pen=="UOT_sinkhorn":
            uy,vy,Gy=prox_UOT_sinkhorn(G.T@u,b,Cy,lam2/eps,eps2,numiter2,lam3,uy)
            Qy=uy.reshape((-1, 1)) * Gy * vy.reshape((1, -1))
            prox=torch.sum(Qy,axis=1)
                                   
        v=prox/(G.T@u)
        
        if pen is None:
            prox=a
        elif pen=="kl":
            prox=prox_KL(G@v,a,lam,eps)
        elif pen=="l2":
            prox=prox_l2(G@v,a,lam)
        if pen=="MMD":
            prox=prox_MMD(G@v,a,Kx,lam,eps,ux,numiter2,lr)
            ux=prox.clone()
        elif pen=="sinkhorn":
            ux,vx,Gx=prox_sinkhorn(G@v,a,Cx,lam/eps,eps2,numiter2,ux)
            Qx=ux.reshape((-1, 1)) * Gx * vx.reshape((1, -1)) 
            prox=torch.sum(Qx,axis=1)
        elif pen=="UOT_sinkhorn":
            ux,vx,Gx=prox_UOT_sinkhorn(G@v,a,Cx,lam/eps,eps2,numiter2,lam3,ux)
            Qx=ux.reshape((-1, 1)) * Gx * vx.reshape((1, -1))
            prox=torch.sum(Qx,axis=1) 
  
        u=prox/(G@v)
    
        if (torch.norm(u_old-u)+torch.norm(v-v_old))<thr:
                if innerplan:
                    return u,v,G,Qx,Qy
                else:
                    return u,v,G
        else:
            u_old=u
            v_old=v
    if innerplan:
        return u,v,G,Qx,Qy
    else:
        return u,v,G


      
def prox_KL(Gu,b,lam,eps):
    gam1=lam/(eps+lam)
    gam2=eps/(eps+lam)
    return (Gu**gam2)*(b**gam1)
    
def prox_l2(Gv,b,lam):
    return (torch.real(lambertw(lam*Gv*torch.exp(lam*b), k=0, tol=1e-8))/lam).float()

def prox_MMD(Gv,b,K,lam,eps,u_warm=None,numiter=100,lr=1e-3):
    if u_warm is None:# warmstart
        u_warm=Gv
    u,loss_l=logGD_MMDpen(Gv,b,K,lam,eps,u_warm,numiter,lr)
    #pl.plot(loss_l)
    return u
def MMDpen(u,Gv,b,K,lam,eps):
    return eps*(torch.sum(u*torch.log(u/Gv)-u+Gv))+lam*(b@K@b+u@K@u-2*b@K@u)


def loggrad_MMD(f,Gv,b,K,lam,eps):
    return eps*(f-torch.log(Gv))+2*lam*K@(torch.exp(f)-b)
  
def logGD_MMDpen(Gv,b,K,lam,eps,u_warm,numiter=100,lr=1e-2):
    f=torch.log(u_warm)
    loss_l=[]
    for i in range(numiter):
        f_next=f-lr*loggrad_MMD(f,Gv,b,K,lam,eps)
        f=f_next.clone()
        loss_l+=[MMDpen(torch.exp(f),Gv,b,K,lam,eps)]
    return torch.exp(f),loss_l
        
def prox_sinkhorn(Gv,b,C,lam,eps2,numiter,u): 
    if u is None:# warmstart
        u=Gv
    G=torch.exp(-C/eps2) #G symmetric matrix since C is
    gamma=lam/(eps2+lam)
    u_old=u
    v_old=b
    for i in range(numiter):
        v=b/(G@u)
        u=(Gv/(G@v))**gamma
        if (torch.norm(u_old-u)+torch.norm(v-v_old))<1e-5:
            v=b/(G@u)
            return u,v,G
        else:
            u_old=u
            v_old=v
    v=b/(G@u)
    return u,v,G
    
def prox_UOT_sinkhorn(Gv,b,C,lam1,eps2,numiter,lam2,u):
    if u is None:
        u=Gv
    G=torch.exp(-C/eps2)#G symmetric matrix since C is
    gamma1=lam1/(eps2+lam1)
    gamma2=lam2/(eps2+lam2)
    u_old=u
    v_old=b
    for i in range(numiter):
        v=(b/(G@u))**gamma2
        u=(Gv/(G@v))**gamma1
        if (torch.norm(u_old-u)+torch.norm(v-v_old))<1e-5:
            v=b/(G@u)
            return u,v,G
        else:
            u_old=u
            v_old=v
    return u,v,G
"""
### GD without log ###
"""def grad_MMDpen(u,Gv,b,K,lam,eps):
    return eps*torch.log(u/Gv)+2*lam*K@(torch.exp(u)-b)
    
def GD_MMDpen(Gv,b,K,lam,eps,u_warm,numiter=100,lr=1e-2):
    u=u_warm
    loss_l=[]
    for i in range(numiter):
        u_next=u-lr*grad_MMDpen(u,Gv,b,K,lam,eps)
        u=u_next.clone()
        loss_l+=[MMDpen(torch.exp(u),Gv,b,K,lam,eps)]
    return u,loss_l"""


    
    
