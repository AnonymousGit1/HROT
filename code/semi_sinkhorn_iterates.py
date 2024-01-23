import torch
import ot
    

def semi_sinkhorn(a,b,C,lam2=50,eps=1,numiter=500,pen=None,
             numiter2=1,Ky=None,lr=1e-3,thr=1e-5):
    u_old=a
    v_old=b
    u=a
    G=torch.exp(-C/eps)
    uy=G.T@u
    ux=None
    for i in range(numiter):
        if pen is None:
            prox=b
        elif pen=="kl":
            prox=prox_KL(G.T@u,b,lam2,eps)
        elif pen=="MMD":
            prox=prox_MMD(G.T@u,b,Ky,lam2,eps,uy,numiter2,lr)
            uy=prox.clone()
                                   
        v=prox/(G.T@u)
        
        u=a/(G@v)
    
        if (torch.norm(u_old-u)+torch.norm(v-v_old))<thr:
            return u,v,G
        else:
            u_old=u
            v_old=v
    return u,v,G

      
def prox_KL(Gu,b,lam,eps):
    gam1=lam/(eps+lam)
    gam2=eps/(eps+lam)
    return (Gu**gam2)*(b**gam1)

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

