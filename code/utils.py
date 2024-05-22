import matplotlib.pylab as pl
import numpy as np
import torch 
import ot
from matplotlib.colors import ListedColormap
from coloraide import Color
from sklearn.cluster import KMeans
    
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
                
def KL(a,b):
    return np.sum(a*np.log(a/b)-a+b)

def compute_kmeans_centroids(X, **kwargs):
    kmeans = KMeans(**kwargs).fit(X)
    return kmeans.cluster_centers_
