import torch
import ot
    

def sinkhorn(a,b,C,lam=50,eps=1,numiter=500,lam2=None,pen=None,
             Cx=None,Cy=None,lam3=None,eps2=None,numiter2=1,innerplan=False,thr=1e-5):
    u_old=a
    v_old=b
    u=a
    G=torch.exp(-C/eps) #Gibbs Kernel
    uy=G.T@u
    ux=None
    if lam2 is None:
        lam2=lam
    if lam3 is None:
        lam3=lam
    if eps2 is None:
        eps2=eps
    if pen =="sinkhorn" or pen=="UOT_sinkhorn":#Gibbs kernel source and target
        Gx=torch.exp(-Cx/eps2)
        Gy=torch.exp(-Cy/eps2)
        
    for i in range(numiter):
        if pen is None:
            prox=b
        elif pen=="kl":
            prox=prox_KL(G.T@u,b,lam2,eps)
        elif pen=="sinkhorn":
            uy,vy=prox_sinkhorn(G.T@u,b,Gy,eps/lam,eps2,numiter2,uy)
            Qy=uy.reshape((-1, 1)) * Gy * vy.reshape((1, -1)) 
            prox=torch.sum(Qy,axis=1)
        elif pen=="UOT_sinkhorn":
            uy,vy=prox_UOT_sinkhorn(G.T@u,b,Gy,eps/lam,eps2,numiter2,lam3,uy)
            Qy=uy.reshape((-1, 1)) * Gy * vy.reshape((1, -1))
            prox=torch.sum(Qy,axis=1)
                                   
        v=prox/(G.T@u)
        
        if pen is None:
            prox=a
        elif pen=="kl":
            prox=prox_KL(G@v,a,lam,eps)
        elif pen=="sinkhorn":
            ux,vx=prox_sinkhorn(G@v,a,Gx,lam/eps,eps2,numiter2,ux)
            Qx=ux.reshape((-1, 1)) * Gx * vx.reshape((1, -1)) 
            prox=torch.sum(Qx,axis=1)
        elif pen=="UOT_sinkhorn":
            ux,vx=prox_UOT_sinkhorn(G@v,a,Gx,lam/eps,eps2,numiter2,lam3,ux)
            Qx=ux.reshape((-1, 1)) * Gx * vx.reshape((1, -1))
            prox=torch.sum(Qx,axis=1) 
  
        u=prox/(G@v)
    
        if (torch.norm(u_old-u)+torch.norm(v-v_old))<thr: #If convergences early stop
                P=u.reshape((-1, 1)) * G * v.reshape((1, -1))
                if innerplan:
                    return P,Qx,Qy
                else:
                    return P
        else:
            u_old=u
            v_old=v
    P=u.reshape((-1, 1)) * G * v.reshape((1, -1))
    if innerplan:
        return P,Qx,Qy
    else:
        return P


      
def prox_KL(Gu,b,lam,eps):
    gam1=lam/(eps+lam)
    gam2=eps/(eps+lam)
    return (Gu**gam2)*(b**gam1)

        
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
    if u is None: #warmstart
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
    

    
    
