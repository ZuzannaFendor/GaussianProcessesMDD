import sklearn as sk

from sklearn.model_selection import TimeSeriesSplit
from AnalysisUtils import run_inference_pcg as rn
import numpy as np
import pandas as pd
from AnalysisUtils import data_processing_pcg as datapcg
from AnalysisUtils import plot_pcg
from AnalysisUtils import run_inference_pcg as runpcg
import tensorflow as tf
import matplotlib.pyplot as plt
from AnalysisUtils import network_pcg
import networkx as nx

def generate_data(N=100):
    X = np.random.rand(N)[:, None] * 10 - 5  # Inputs = N x D
    G = np.hstack((0.5 * np.sin(3 * X) + X, 3.0 * np.cos(X) - X))  # G = N x L
    W = np.array([[0.5, -0.3, 1.5, 0.2, 1.1], [-0.4, 0.43, 0.0, 1.2,-0.1]])  # L x P
    F = np.matmul(G, W)  # N x P
    Y = F + np.random.randn(*F.shape) * [0.2, 0.2, 0.2,0.2,0.2]

    return X, Y
Xgen,Ygen = datagen = generate_data()
data,col_names, _  = datapcg.import_ESMdata()


print(data.head(5))
m1 = runpcg.run_example((Xgen,Ygen))
Xgen = Xgen[:,0]
m2 = runpcg.run_MOGP((Xgen,Ygen), iterations = 100)


X = data['hour_no'].to_numpy(dtype="float64")[:100][:,None]
Y = data.loc[:,['neg_affect','pos_affect','sus','worry','mental_unrest']].to_numpy(dtype="float64")[:100,:] #,'worry','mental_unrest'
N, D, T = Y.shape[0],Y.shape[1], int(np.max(X))

num_iter = 20000
num_samples = 200
N_test = 300
testpoints = np.linspace(0, T, N_test)
tiled_testpoints = np.tile(testpoints, (D, 1)).T
num_inducing = int(N*0.3)
runpcg.run_example((X,Y), lower = np.min(X), upper = np.max(X))
# mu_y,var_y = mogp_model.predict_f(np.array(testpoints))
# mumu = np.reshape(mu_y,(N_test,D)).T
# vuvu = np.reshape(var_y,(N_test,D)).T
#
# plot_pcg.plot_timeseries(testpoints, mumu, vuvu, X, Y)