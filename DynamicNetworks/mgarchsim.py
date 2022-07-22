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
period = 0.7 # the period of the signal
N = 500 #number of data points total
N_test = 40 # number of test data
D = 3 # number of output dimensions
n_sim = 8# number of data simulations
s =150 #number of y samples given mu and sigma
sim_type = "periodic"
mgarch_sigma_list = np.zeros((n_sim, N, D,D))
mgarch_mu_pred = np.zeros((n_sim, N_test, D))

mgarch_y_pred = np.zeros((n_sim, s, N_test, D))
mgarch_cov_pred = np.zeros((n_sim,N_test,D,D))
ymse_mgarch_list =np.zeros((n_sim,s, N_test,D))

correlations = []
loglik_mgarch_list = np.zeros((n_sim,N_test))

for i in range(n_sim):
    #create dataset
    data, Ks, Sigmas = datapcg.simulate_data(duration, N, size=3, period=period, type="periodic", max=0.7, min=0.3)
    X = np.linspace(0, duration, N)
    Y = data

    #train model
    #mgarch
    mgarch_model = runpcg.run_MGARCH((X, Y), refit_every = 5, nrTest = N_test)
    mgarch_sigma_list[i] = mgarch_model["covariance_matrix"]
    mgarch_mu_pred[i] = mgarch_model["mu_predictions"]
    mgarch_cov_pred[i] = mgarch_model["cov_predictions"]

    #compute log likelihood
    loglik_mgarch_list[i] = metrics.log_lik(data[-N_test:], mgarch_cov_pred[i], mgarch_mu_pred[i])

    # use the conditional mu and sigma to sample data points
    ymu, yvar, y_samples = runpcg.sample_y(np.tile(mgarch_mu_pred[i], (s,1,1)), np.tile(mgarch_cov_pred[i], (s,1,1,1)))
    mgarch_y_pred[i] = y_samples

    # output prediction mean squared error
    ymse_mgarch_list[i] = metrics.MSE(data[-N_test:], y_samples)

    #save covariance plot
    plot_pcg.plot_cov_comparison(X, Sigmas, np.reshape(mgarch_sigma_list[i], (1, N, D, D)), D, "MGARCH",pred = np.reshape(mgarch_cov_pred[i],(1,N_test,D,D)), save=f"mgarch_simulation/mgarch{i}", lim=(-4,6))
    np.save(f"mgarch_simulation/data{i}_{sim_type}", data)
    np.save(f"mgarch_simulation/trueSigma{i}_{sim_type}", Sigmas)
    np.save(f"mgarch_simulation/trueKs{i}_{sim_type}", Ks)
plot_pcg.plot_cov_comparison(X, Sigmas, mgarch_sigma_list, D, "MGARCH",pred = mgarch_cov_pred, save=f"mgarch_simulation/mgarch{sim_type} average", lim = (-4,6.5))

#save metrics for cov-cov
np.save(f"mgarch_simulation/sigmas_{sim_type}",mgarch_sigma_list)
np.save(f"mgarch_simulation/mus_{sim_type}", mgarch_mu_pred)
np.save(f"mgarch_simulation/ysamples_{sim_type}",mgarch_y_pred)

# sigma prediciton mean squared error
mse_mgarch = metrics.MSE(Sigmas[-N_test:], mgarch_cov_pred)
np.save(f"mgarch_simulation/mse_simulation_{sim_type}_cov",mse_mgarch)

#save metrics for y-y
np.save(f"mgarch_simulation/ymse_simulation_{sim_type}_cov",ymse_mgarch_list)

correlations_loglik = np.ones((n_sim))
for i in range(n_sim):
    avgmse = np.mean(np.mean(mse_mgarch[i], axis = -1), axis = -1)
    ll = loglik_mgarch_list[i]
    correlations_loglik[i],_ = metrics.corr_timeseries(avgmse,ll)


plt.hist(correlations_loglik, bins =20)
plt.xlabel("correlation")
plt.ylabel("count")
plt.title("Correlation between Sigma MSE and the log-likelihood")
plt.savefig(f"mgarch_simulation/{sim_type}likelihoodcorr")
plt.close()


#sigma prediction for the test data
#averaged over the output dimensions
# mse = np.mean(np.reshape(np.triu(mse_mgarch[:, -N_test:]), (n_sim, N_test, D * D)), axis=-1)
# ymse = np.mean(ymse_mgarch_list, axis=-1)
correlations = np.zeros((n_sim,s, D))
pvalues = np.zeros_like(correlations)
for i in range(n_sim):
    for j in range(s):
        for d in range(D):
            ah = metrics.corr_timeseries(np.mean(mse_mgarch[i,:,d], axis = -1),ymse_mgarch_list[i,j,:,d])
            correlations[i,j,d], pvalues[i,j,d] = ah

#plot the distribution of the correlations
#based on all samples collected
c = np.reshape(correlations, (n_sim*s*D))
plt.hist(c, bins =20)
plt.title("Correlation between Sigma MSE and Y MSE")
plt.xlabel("correlation")
plt.ylabel("count")
plt.savefig(f"mgarch_simulation/{sim_type}msecorr")
plt.close()
#plot the distribution of the correlations
#per dimension
c = np.reshape(correlations, (n_sim*s,D))
for d in range(D):
    plt.hist(c[:,d], bins =20)
    plt.title(f"mgarch_simulation/Correlation histogram of dimension {d}")
    plt.show()
plt.close()

#plot significance and correlation for different simulation data
indices = np.argsort(np.reshape(pvalues, (n_sim*s*D)))
c = np.reshape(correlations, (n_sim*s*D))
z = np.take(c, indices)
plt.scatter(np.take(pvalues, indices),z)
plt.title("correlation against significance")
plt.ylabel("correlation")
plt.xlabel("p-value")
plt.show()
##### correlation check ######

#compute the correlation for sigma-sigma and y-prediction y
ycor = np.zeros((n_sim,s,D))
sigmacor = np.zeros((n_sim,D,D))
for i in range(n_sim):
    data = np.load(f"mgarch_simulation/data{i}_{sim_type}.npy")
    for d in range(D):
        for sim in range(s):
            ycor[i,sim,d],_ = metrics.corr_timeseries(data[-N_test:,d], mgarch_y_pred[i,sim,:,d])
        for e in range(D):
            sigmacor[i,d,e],_ =metrics.corr_timeseries(Sigmas[-N_test:,d,e], mgarch_cov_pred[i,:,d,e])

#compute correlations between correlations
correlations_b = np.zeros((n_sim, s, d))
pvalues_b = np.zeros((n_sim, s, d))
# since the correlation gives the time series
# a single value, we do not have enough values to perform
# correlation analysis between correlation of sigmas and
# correlation of ys. For this reason, we will not obtain a
# distribution fo correlations like in the previous case.
# We will switch it up and compute

# ycornew = np.mean(np.mean(ycor, axis = -1), axis = -1)
# sgigmacornew = np.mean(np.mean(sigmacor, axis = -1), axis = -1)
# general_correlation, general_pvalue = scipy.stats.pearsonr(ycornew, sgigmacornew)
# print("gen correlation correlation",general_correlation)

print("output prediction metrics:")
print("average likelihood y|sigma, mu:", np.mean(loglik_mgarch_list))
print("average mse y vs ypred",np.mean(ymse_mgarch_list))
print("average correlation y with ypred:",np.mean(ycor))

print("sigma vs true sigma metrics")
print("average correlation sigma with sigma pred:",np.mean(sigmacor))
print("average mse sigma with sigma pred:",np.mean(mse_mgarch))

print("mean correlation log likelihood and mse sigma",np.mean(correlations_loglik))
print("mean mse corr", np.mean(correlations))



print("end")
print("security")