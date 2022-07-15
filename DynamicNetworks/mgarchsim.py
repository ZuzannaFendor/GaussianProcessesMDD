import scipy.stats
import sklearn as sk

from sklearn.model_selection import TimeSeriesSplit
from AnalysisUtils import run_inference_pcg as rn
import numpy as np
import pandas as pd
from AnalysisUtils import data_processing_pcg as datapcg
from AnalysisUtils import plot_pcg
from AnalysisUtils import run_inference_pcg as runpcg
from AnalysisUtils import metrics
import tensorflow as tf
import matplotlib.pyplot as plt
from AnalysisUtils import network_pcg
import networkx as nx
import gpflow
from Banner.util import training_util


duration = 6 #the time span from 0 to x of the dataset
period = 1.7 # the period of the signal
N = 500 #number of data points total
N_test = 40 # number of test data
D = 3 # number of output dimensions
n_sim = 4# number of data simulations
s =150 #number of y samples given mu and sigma
sim_type = "periodic"
mgarch_sigma_list = np.zeros((n_sim, N, D,D))
mgarch_mu_pred = np.zeros((n_sim, N_test, D))

mgarch_y_pred = np.zeros((n_sim, s, N_test, D))
ymse_mgarch_list =np.zeros((n_sim,s, N_test,D))

correlations = []

for i in range(n_sim):
    #create dataset
    data, Ks, Sigmas = datapcg.simulate_data(duration, N, size=3, period=period, type="periodic", max=0.7, min=0.3)
    X = np.linspace(0, duration, N)
    Y = data

    #train model
    #mgarch
    mgarch_model = runpcg.run_MGARCH((X, Y), refit_every = 5, nrTest = N_test)
    mgarch_sigma_list[i] = mgarch_model["covariance_matrix"]
    mgarch_mu_pred[i] = mgarch_model["y_predictions"]

    # use the conditional mu and sigma to sample data points
    ymu, yvar, y_samples = runpcg.sample_y(np.tile(mgarch_mu_pred[i], (s,1,1)), np.tile(mgarch_sigma_list[i,-N_test:], (s,1,1,1)))
    mgarch_y_pred[i] = y_samples

    # output prediction mean squared error
    ymse_mgarch_list[i] = metrics.MSE(data[-N_test:], y_samples)

    #save covariance plot
    plot_pcg.plot_cov_comparison(X, Sigmas, np.reshape(mgarch_sigma_list[i], (1, N, D, D)), D, "MGARCH", save=f"mgarch_simulation/mgarch{i}", lim=(-4,6))
    np.save(f"mgarch_simulation/data{i}_{sim_type}", data)

#save metrics for cov-cov
np.save("mgarch_simulation/sigmas_periodic",mgarch_sigma_list)
np.save("mgarch_simulation/mus_periodic", mgarch_mu_pred)
np.save("mgarch_simulation/ysamples_periodic",mgarch_y_pred)

# sigma prediciton mean squared error
mse_mgarch = metrics.MSE(Sigmas, mgarch_sigma_list)
np.save("mgarch_simulation/mse_simulation_periodic_cov",mse_mgarch)

#save metrics for y-y
np.save("mgarch_simulation/ymse_simulation_periodic_cov",ymse_mgarch_list)

#sigma prediction for the test data
#averaged over the output dimensions
# mse = np.mean(np.reshape(np.triu(mse_mgarch[:, -N_test:]), (n_sim, N_test, D * D)), axis=-1)
# ymse = np.mean(ymse_mgarch_list, axis=-1)
for i in range(n_sim):
    for j in range(s):
        correlations.append([metrics.corr_timeseries(np.mean(mse_mgarch[i,-N_test:,d], axis = -1),ymse_mgarch_list[i,j,:,d]) for d in range(D)])

#plot the distribution of the correlations
c = np.array(correlations)
plt.hist(c[:,0], bins =15)
plt.show()
plt.plot()

#plot significance and correlation for different simulation data
z = c[c[:, 1].argsort()]
plt.scatter(z[:,1],z[:,0])
plt.title("correlation against significance")
plt.ylabel("correlation")
plt.xlabel("p-value")

#compute the correlation for sigma-sigma and y-prediction y
ycor = np.zeros((n_sim,D))
sigmacor = np.zeros((n_sim,D,D))
for i in range(n_sim):
    data = np.load(f"mgarch_simulation/data{i}_{sim_type}.npy")
    for d in range(D):
        ycor[i,d],_ = metrics.corr_timeseries(data[-N_test:,d], np.mean(mgarch_y_pred[i], axis = 0)[:,d])
        for e in range(d,D):
            sigmacor[i,d,e],_ =metrics.corr_timeseries(Sigmas[:,d,e], mgarch_sigma_list[i,:,d,e])

#compute correlations between correlations
correlations_b = []
for i in range(n_sim):
    correlations_b.append(metrics.corr_timeseries(ycor[i],sigmacor[i]))

#plot the distribution of the correlations
c = np.array(correlations_b)
plt.hist(c[:,0], bins =15)
plt.show()
plt.plot()

#plot significance and correlation for different simulation data
z = c[c[:, 1].argsort()]
plt.scatter(z[:,1],z[:,0])
plt.title("correlation against significance")
plt.ylabel("correlation")
plt.xlabel("p-value")

#plot mean squared error over time
tf.math.reduce_mean(mse, axis  = 0).numpy()
tf.math.reduce_variance(mse, axis=0).numpy()
plot_pcg.plot_timeseries(X,ymu, yvar, np.array([0]), np.array([0]))

print("end")
print("security")