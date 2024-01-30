import numpy as np
from ripser import ripser
from scipy import sparse



#Inspired from the code provides with the article: Topological Descriptors for Parkinson’s Disease Classification and Regression Analysis
# https://github.com/itsmeafra/Sublevel-Set-TDA
        
def curve_to_matrix(x):
        
    # Extracting Needed Info
    N = len(x)
    t = np.arange(N)
    
    #Sublevelset Filtration
    # Add edges between adjacent points in the time series, with the "distance"
    # along the edge equal to the max value of the points it connects
    I = np.arange(N-1)
    J = np.arange(1, N)
    V = np.maximum(x[0:-1], x[1::])
    
    # Add vertex birth times along the diagonal of the distance matrix
    I = np.concatenate((I, np.arange(N)))
    J = np.concatenate((J, np.arange(N)))
    V = np.concatenate((V, x))
    
    #Create the sparse distance matrix
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    
    return D

def curve_to_diagram(x):
        
    # Extracting Needed Info
    N = len(x)
    t = np.arange(N)
    
    #Sublevelset Filtration
    # Add edges between adjacent points in the time series, with the "distance"
    # along the edge equal to the max value of the points it connects
    I = np.arange(N-1)
    J = np.arange(1, N)
    V = np.maximum(x[0:-1], x[1::])
    
    # Add vertex birth times along the diagonal of the distance matrix
    I = np.concatenate((I, np.arange(N)))
    J = np.concatenate((J, np.arange(N)))
    V = np.concatenate((V, x))
    
    #Create the sparse distance matrix
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    dgm0 = ripser(D, maxdim=0, distance_matrix=True)['dgms'][0]
    dgm0 = dgm0[dgm0[:, 1]-dgm0[:, 0] > 1e-3, :]
    allgrid = np.unique(dgm0.flatten())
    allgrid = allgrid[allgrid < np.inf]
    xs = np.unique(dgm0[:, 0])
    ys = np.unique(dgm0[:, 1])
    ys = ys[ys < np.inf]
    
    # Removing Infinity Points
    dgm0[np.isinf(dgm0)]=np.max(x)
    
    return dgm0
