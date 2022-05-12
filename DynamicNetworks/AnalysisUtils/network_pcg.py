import numpy as np
import scipy as sci
import networkx as nx
import sklearn
from sklearn.covariance import GraphicalLassoCV
import random
import matplotlib.pyplot as plt

from matplotlib import animation


def partial_correlations(K):
    '''
    param K: RxTxDxD matrix of covariances
            R: number of samples
            T: number of time steps
            D: number of output dimensions
    returns par_cor: RxTxDxD  partial correlation matrix
    '''
    par_cor = np.ones_like(K)

    n_sample = K.shape[0]
    n_time = K.shape[1]
    n_dim = K.shape[2]
    for s in range(n_sample):
        for t in range(n_time):
            for i in range(n_dim):
                for j in range(i + 1, n_dim):
                    if i != j:
                        par_cor[s, t, i, j] = -K[s, t, i, j] / (np.sqrt(K[s, t, i, i]) * np.sqrt(K[s, t, j, j]))
                        par_cor[s, t, j, i] = par_cor[s, t, i, j]
    return par_cor