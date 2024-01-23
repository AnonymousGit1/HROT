import matplotlib.pylab as pl
import numpy as np
import torch 
import ot
from matplotlib.colors import ListedColormap
from coloraide import Color

def kernel(X,Y,sigma=5,k="gaussian"):
    if k=="gaussian":
        Kx=torch.exp(-ot.dist(X,X)/(2*sigma**2))
        Ky=torch.exp(-ot.dist(Y,Y)/(2*sigma**2))
    if k=="laplace1":
        Kx=torch.tensor(np.exp(-ot.dist(X.numpy(),X.numpy(),metric='cityblock')/(sigma))).float()
        Ky=torch.tensor(np.exp(-ot.dist(Y.numpy(),Y.numpy(),metric='cityblock')/(sigma))).float()
    if k=="laplace2":
        Kx=torch.exp(-ot.dist(X,X,metric="euclidean")/sigma)
        Ky=torch.exp(-ot.dist(Y,Y,metric="euclidean")/sigma)
    if k=="energy":
        Kx=-ot.dist(X,X,metric='euclidean')
        Ky=-ot.dist(Y,Y,metric='euclidean')
    if k=="power":
        Kx=torch.exp(-(torch.tensor(ot.dist(X.numpy(),X.numpy(),metric='minkowski', p=3))**3)/(2*sigma**2))
        Ky=torch.exp(-(torch.tensor(ot.dist(Y.numpy(),Y.numpy(),metric='minkowski', p=3))**3)/(2*sigma**2))
    return Kx,Ky
    
def plot2D_plan(xs, xt, G, thr=1e-8, **kwargs):
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    mx = G.max()
    if mx<1:
        mx=1
    if 'alpha' in kwargs:
        scale = kwargs['alpha']
        del kwargs['alpha']
    else:
        scale = 1
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                pl.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                        alpha=G[i, j] / mx * scale, **kwargs)
                        
def plot2D_plan_vocab(xs, xt, G,ssize=100, thr=1e-8, **kwargs):
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    mx = G.max()
    if mx<1:
        mx=1
    if 'alpha' in kwargs:
        scale = kwargs['alpha']
        del kwargs['alpha']
    else:
        scale = 1
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                pl.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],linewidth=ssize*G[i, j] / mx, **kwargs)
                
def def_colormap():                
    inter = Color.interpolate(["rebeccapurple", "rgb(255,202,0)"],space='srgb')
    vals = np.ones((256, 4))
    for i in range(256):
        col=inter(i/256)
        vals[i,0]=col[0]
        vals[i,1]=col[1]
        vals[i,2]=col[2]

    cmp = ListedColormap(vals)
    return cmp
