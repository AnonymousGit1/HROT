import numpy as np
from numpy import linalg as LA

def LOT_freesupp(source, target,n_source_anchors,n_target_anchors,
        epsilon=1, epsilon_z=1, intensity=[10, 10, 10], niter=500,
        tolratio=1e-7, p=2, random_state=None):
    
    # centroid initialized by K-means
    Cx = compute_kmeans_centroids(source, n_clusters=n_source_anchors, random_state=random_state)
    Cy = compute_kmeans_centroids(target, n_clusters=n_target_anchors, random_state=random_state)
    
    # Px, Py initialized by K-means and one-sided OT
    n = source.shape[0]
    m = target.shape[0]
    mu = 1 / n * np.ones([n, 1])
    nu = 1 / m * np.ones([m, 1])
    cost_xy = compute_cost_matrix(source, target, p=p)
    P = np.zeros([n,m]) + 1 / n / m

    converrlist = np.zeros(niter) + np.inf
    
    for t in range(0, niter):
        # compute cost matrices
        cost_x = compute_cost_matrix(source, Cx, p=p)
        cost_z = compute_cost_matrix(Cx, Cy, p=p)
        cost_y = compute_cost_matrix(Cy, target, p=p)
        
        Kx = np.exp(-intensity[0] * cost_x / epsilon)
        Kz = np.exp(-intensity[1] * cost_z / epsilon_z)
        Ky = np.exp(-intensity[2] * cost_y / epsilon)
            
        Pt1 = P
        Px, Py, Pz, P = LOT_fixsupp(Kx, Kz, Ky,epsilon=epsilon,epsilon1=epsilon_z)  # update trans. plan

        # check for convergence
        converr = LA.norm(P - Pt1) / LA.norm(Pt1)
        converrlist[t] = converr
        if converr < tolratio:
            break

        # update anchors
        if t < niter - 1:
            Cx, Cy = update_anchors(Px, Py, Pz, source, target,n_source_anchors,n_target_anchors,intensity)
    return Cx, Cy,Px, Py, Pz, P

def LOT_fixsupp(Kx, Kz, Ky,mu,nu, niter=100, tol=1e-5, epsilon=0, clip_val=np.inf, epsilon1 = 0):
    dimx = Kx.shape[0]
    dimy = Ky.shape[1]
    dimz1, dimz2 = Kz.shape

    ax=np.ones((dimx,))
    bx=np.ones((dimz1,))
    ay=np.ones((dimz2,))
    by=np.ones((dimz1,))
    az=np.ones((dimz1,))
    bz=np.ones((dimz2,))
    wxz=np.ones((dimz1,))
    wzy=np.ones((dimz2))
    
    for i in range(1, niter + 1):   
        ax = np.exp(np.minimum(np.log(np.maximum(mu,epsilon1)) - np.log(np.maximum(Kx.dot(bx), epsilon1)), clip_val))
        err1x = LA.norm(bx * Kx.T.dot(ax) - wxz, ord=1)
            

        by = np.exp(np.minimum(np.log(np.maximum(nu,epsilon1)) - np.log(np.maximum(Ky.T.dot(ay), epsilon1)), clip_val))
        err2y = LA.norm(ay * (Ky.dot(by)) - wzy, ord=1)
            
               
        wxz = ((az * (Kz.dot(bz))) * (bx * (Kx.T.dot(ax)))) ** (1 / 2)
        bx = np.exp(np.minimum(np.log(np.maximum(wxz, epsilon)) - np.log( np.maximum(Kx.T.dot(ax),epsilon)), clip_val))
        err2x = LA.norm(ax * (Kx.dot(bx)) - mu, ord=1)

        az = np.exp(np.minimum(np.log(np.maximum(wxz, epsilon)) - np.log(np.maximum(Kz.dot(bz), epsilon)), clip_val))
        err1z = LA.norm(bz * Kz.T.dot(az) - wzy, ord=1)
        wzy = ((ay * (Ky.dot(by))) * (bz * (Kz.T.dot(az)))) ** (1 / 2)
        bz = np.exp(np.minimum(np.log(np.maximum(wzy,epsilon)) - np.log(np.maximum(Kz.T.dot(az), epsilon)), clip_val))
        err2z = LA.norm(az * (Kz.dot(bz)) - wxz, ord=1)

        ay = np.exp(np.minimum(np.log(np.maximum(wzy, epsilon)) - np.log(np.maximum(Ky.dot(by), epsilon)), clip_val))
        err1y = LA.norm(by * Ky.T.dot(ay) - nu, ord=1)
        #if i==niter:
        #    print(max(err1x, err2x, err1z, err2z, err1y, err2y))
        if max(err1x, err2x, err1z, err2z, err1y, err2y) < tol:
            break

    Px=ax.reshape((-1, 1)) * Kx * bx.reshape((1, -1)) 
    Py=ay.reshape((-1, 1)) * Ky * by.reshape((1, -1)) 
    Pz=az.reshape((-1, 1)) * Kz * bz.reshape((1, -1)) 
    const = 0
    z1 = Px.T.dot(np.ones([dimx, 1])) + const
    z2 = Py.dot(np.ones([dimy, 1])) + const
    P = np.dot(Px / z1.T, np.dot(Pz, Py / z2))
    return Px, Py, Pz, P

def update_anchors(Px, Py, Pz, source, target,n_source_anchors,n_target_anchors,intensity=[10, 10, 10]):
    n = source.shape[0]
    m = target.shape[0]
    Px = intensity[0] * Px
    Pz = intensity[1] * Pz
    Py = intensity[2] * Py

    temp = np.concatenate((np.diagflat(Px.T.dot(np.ones([n, 1])) +
                                           Pz.dot(np.ones([n_target_anchors, 1]))), -Pz), axis=1)
    temp1 = np.concatenate((-Pz.T, np.diagflat(Py.dot(np.ones([m, 1])) +
                                                   Pz.T.dot(np.ones([n_source_anchors, 1])))), axis=1)
    temp = np.concatenate((temp, temp1), axis=0)
    sol = np.concatenate((source.T.dot(Px), target.T.dot(Py.T)), axis=1).dot(LA.inv(temp))
    Cx = sol[:, 0:n_source_anchors].T
    Cy = sol[:, n_source_anchors:n_source_anchors + n_target_anchors].T
    return Cx, Cy

def compute_cost_matrix(source, target, p=2):
    cost_matrix = np.sum(np.power(source.reshape([source.shape[0], 1, source.shape[1]]) -
                                  target.reshape([1, target.shape[0], target.shape[1]]),
                                  p), axis=-1)
    return cost_matrix


"""""""""""""""
"""""""""""""""

def transport(Px,Py,Pz,source, target,n_source_anchors,n_target_anchors):
    n = source.shape[0]
    m = target.shape[0]
    Cx_lot = Px_.T.dot(source) / (Px_.T.dot(np.ones([n, 1])) + 10 ** -20)
    Cy_lot = Py_.dot(target) / (Py_.dot(np.ones([m, 1])) + 10 ** -20)
    transported = source + np.dot(
            np.dot(Px_ / np.sum(Px_, axis=1).reshape([n, 1]),Pz_ / np.sum(Pz_, axis=1).reshape([n_source_anchors, 1])
            ),
            Cy_lot) - np.dot(Px_ / np.sum(Px_, axis=1).reshape([n, 1]), Cx_lot)
    return transported

def robust_transport(Px,Py,Pz,source, target,n_source_anchors,n_target_anchors, threshold=0.8, decay=0):
    n = source.shape[0]
    m = target.shape[0]
    Cx_lot = Px_.T.dot(source) / (Px_.T.dot(np.ones([n, 1])) + 10 ** -20)
    Cy_lot = Py_.dot(target) / (Py_.dot(np.ones([m, 1])) + 10 ** -20)

    maxPz = np.max(Pz_, axis=1)
    Pz_robust = Pz_.copy()

    for i in range(0, n_source_anchors):
        for j in range(0, n_target_anchors):
            if Pz_[i, j] < maxPz[i] * threshold:
                Pz_robust[i, j] = Pz_[i, j] * decay
    Pz_robust = Pz_robust / np.sum(Pz_robust, axis=1).reshape([n_source_anchors, 1]) * np.sum(Pz_, axis=1).reshape([n_source_anchors, 1])
    transported = source + np.dot(
            np.dot(Px_ / np.sum(Px_, axis=1).reshape([n, 1]),
                Pz_robust / np.sum(Pz_robust, axis=1).reshape([n_source_anchors, 1])
            ), Cy_lot) - np.dot(Px_ / np.sum(Px_, axis=1).reshape([n, 1]), Cx_lot)
    return transported
