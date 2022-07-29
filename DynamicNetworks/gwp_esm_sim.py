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
N_test = 40 # number of test data 40
#retrain every 5 points
D = 3 # number of output dimensions
n_sim = 5# number of data simulations 8
S =150 #number of y samples given mu and sigma
sim_type = "periodic"
gwp_sigma_list = np.zeros((n_sim, S, N, D, D))
gwp_mu_pred = np.zeros((n_sim, S, N_test, D))

gwp_y_pred = np.zeros((n_sim, S, N_test, D))
gwp_cov_pred = np.zeros((n_sim, S, N_test, D, D))
ymse_gwp_list =np.zeros((n_sim, S, N_test, D))

correlations = []
loglik_gwp_list = np.zeros((n_sim, N_test))
num_iter = 65000
batch_size = 100
indu_rate = 0.2

for i in range(n_sim):
    #create dataset
    data = np.load(f"mgarch_simulation/data{i}_{sim_type}.npy")
    Sigmas = np.load(f"mgarch_simulation/trueSigma{i}_{sim_type}.npy")
    X = np.linspace(0, duration, N)
    Y = data

    #train model
    #train GWP
    for it,t in enumerate(range(N-N_test,N,5)):
        print(t)
        x = X[:t]
        y = Y[:t]
        ytest = Y[t:t+5]
        xtest = np.tile(X[t:t+5], (D, 1)).T
        if t == N-N_test:
            wishart_model = runpcg.run_BANNER(data=(x, y), T=duration, mnu = "independent ",l_scale = 1.0, iterations=num_iter,num_inducing=int(indu_rate*N),batch_size=batch_size)
            posterior_wishart_process = wishart_model['wishart process']
        else:
            x = np.tile(x, (D, 1)).T
            datagwp = (x, y)
            elbo = training_util.run_adam(posterior_wishart_process,datagwp,iterations=5000,minibatch_size=batch_size)
        a, b  = posterior_wishart_process.predict_mc(xtest, S)
        gwp_cov_pred[i,:,it*5:it*5+5] , gwp_mu_pred[i,:,it*5:it*5+5] = posterior_wishart_process.predict_mc(xtest, S)
        _, _, gwp_y_pred[i] = runpcg.sample_y( gwp_mu_pred[i],gwp_cov_pred[i] )

    #compute the total cov prediction (given all the data)
    gwp_sigma_list[i], _ = posterior_wishart_process.predict_mc(np.tile(X,(D,1)).T, S)

    #compute log likelihood
    loglik_gwp_list[i] = metrics.log_lik(Y[-N_test:], gwp_cov_pred[i], gwp_mu_pred[i])

    # use the conditional mu and sigma to sample data points

    # output prediction mean squared error
    ymse_gwp_list[i] = metrics.MSE(data[-N_test:], gwp_y_pred[i])

    #save covariance plot
    plot_pcg.plot_cov_comparison(X, Sigmas, gwp_sigma_list[i], D, "GWP", pred = gwp_cov_pred[i], save=f"gwp_simulation/gwp{i}", lim=(-4, 6))

plot_pcg.plot_cov_comparison(X, Sigmas, np.reshape(gwp_sigma_list, (n_sim * S, N, D, D)), D, "GWP", pred = np.reshape(gwp_cov_pred, (n_sim * S, N_test, D, D)), save=f"gwp_simulation/gwp{sim_type} average", lim = (-4, 6.5))

#save metrics for cov-cov
np.save(f"gwp_simulation/sigmas_{sim_type}", gwp_sigma_list)
np.save(f"gwp_simulation/mus_{sim_type}", gwp_mu_pred)
np.save(f"gwp_simulation/ysamples_{sim_type}", gwp_y_pred)

# sigma prediciton mean squared error
mse_gwp = metrics.MSE(Sigmas[-N_test:], np.reshape(gwp_cov_pred, (n_sim * S, N_test, D, D)))
np.save(f"gwp_simulation/mse_simulation_{sim_type}_cov", mse_gwp)

#save metrics for y-y
np.save(f"gwp_simulation/ymse_simulation_{sim_type}_cov", ymse_gwp_list)

correlations_loglik = np.ones((n_sim, S))
mse_resh =np.reshape(gwp_cov_pred, (n_sim , S, N_test, D, D))
for i in range(n_sim ):
    avgmse = np.mean(np.mean(mse_resh, axis = -1), axis = -1)
    ll = loglik_gwp_list[i]
    for s in range(S):
        correlations_loglik[i,s],_ = metrics.corr_timeseries(avgmse[i,s],ll)

correlations_loglik = np.reshape(correlations_loglik, (n_sim*S))
plt.hist(correlations_loglik, bins =20)
plt.title("Correlation between Sigma MSE and the log-likelihood")
plt.xlabel("correlation")
plt.ylabel("count")
plt.savefig(f"gwp_simulation/{sim_type}likelihoodcorr")
plt.close()
#sigma prediction for the test data
#averaged over the output dimensions
# mse = np.mean(np.reshape(np.triu(mse_mgarch[:, -N_test:]), (n_sim, N_test, D * D)), axis=-1)
# ymse = np.mean(ymse_mgarch_list, axis=-1)
correlations = np.zeros((n_sim, S, D))
pvalues = np.zeros_like(correlations)
mse_gwp_reshaped = np.reshape(mse_gwp, (n_sim, S , N_test , D , D))
for i in range(n_sim):
    for j in range(S):
        for d in range(D):
            ah = metrics.corr_timeseries(np.mean(mse_gwp_reshaped[i, j, :, d], axis = -1), ymse_gwp_list[i, j, :, d])
            correlations[i,j,d], pvalues[i,j,d] = ah

#plot the distribution of the correlations
#based on all samples collected
c = np.reshape(correlations, (n_sim * S * D))
plt.hist(c, bins =20)
plt.title("Correlation between Sigma MSE and Y MSE")
plt.xlabel("correlation")
plt.ylabel("count")
plt.savefig(f"gwp_simulation/{sim_type}msecorr")
plt.close()
#plot the distribution of the correlations
#per dimension
c = np.reshape(correlations, (n_sim * S, D))
for d in range(D):
    plt.hist(c[:,d], bins =20)
    plt.title(f"gwp_simulation/Correlation histogram of dimension {d}")
    plt.show()
plt.close()

#plot significance and correlation for different simulation data
indices = np.argsort(np.reshape(pvalues, (n_sim * S * D)))
c = np.reshape(correlations, (n_sim * S * D))
z = np.take(c, indices)
plt.scatter(np.take(pvalues, indices),z)
plt.title("correlation against significance")
plt.ylabel("correlation")
plt.xlabel("p-value")
plt.show()
##### correlation check ######

#compute the correlation for sigma-sigma and y-prediction y
ycor = np.zeros((n_sim, S, D))
sigmacor = np.zeros((n_sim,D,D))
for i in range(n_sim):
    data = np.load(f"mgarch_simulation/data{i}_{sim_type}.npy")
    for d in range(D):
        for sim in range(S):
            ycor[i,sim,d],_ = metrics.corr_timeseries(data[-N_test:,d], gwp_y_pred[i, sim, :, d])
        for e in range(D):
            sigmacor[i,d,e],_ =metrics.corr_timeseries(Sigmas[-N_test:,d,e], np.mean(gwp_cov_pred, axis = 1)[i, :,d, e])

#compute correlations between correlations
correlations_b = np.zeros((n_sim, S, d))
pvalues_b = np.zeros((n_sim, S, d))
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
print("average likelihood y|sigma, mu:", np.mean(loglik_gwp_list))
print("average mse y vs ypred", np.mean(ymse_gwp_list))
print("average correlation y with ypred:",np.mean(ycor))

print("sigma vs true sigma metrics")
print("average correlation sigma with sigma pred:",np.mean(sigmacor))
print("average mse sigma with sigma pred:", np.mean(mse_gwp))

print("mean correlation log likelihood and mse sigma",np.mean(correlations_loglik))
print("mean sigma mse vs y mse corr", np.mean(correlations))



print("end")
print("security")


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
N_test = 40 # number of test data 40
#retrain every 5 points
D = 3 # number of output dimensions
n_sim = 5# number of data simulations 8
S =150 #number of y samples given mu and sigma
sim_type = "linear"
gwp_sigma_list = np.zeros((n_sim, S, N, D, D))
gwp_mu_pred = np.zeros((n_sim, S, N_test, D))


gwp_y_pred = np.zeros((n_sim, S, N_test, D))
gwp_cov_pred = np.zeros((n_sim, S, N_test, D, D))
ymse_gwp_list =np.zeros((n_sim, S, N_test, D))

correlations = []
loglik_gwp_list = np.zeros((n_sim, N_test))
num_iter = 65000
batch_size = 100
indu_rate = 0.2

for i in range(n_sim):
    #create dataset
    data = np.load(f"mgarch_simulation/data{i}_{sim_type}.npy")
    Sigmas = np.load(f"mgarch_simulation/trueSigma{i}_{sim_type}.npy")
    X = np.linspace(0, duration, N)
    Y = data

    #train model
    #train GWP
    for it, t in enumerate(range(N-N_test,N,5)):
        print(t)
        x = X[:t]
        y = Y[:t]
        ytest = Y[t:t+5]
        xtest = np.tile(X[t:t+5], (D, 1)).T
        if t == N-N_test:
            wishart_model = runpcg.run_BANNER(data=(x, y), T=duration, mnu = "independent",l_scale = 1.0, iterations=num_iter,num_inducing=int(indu_rate*N),batch_size=batch_size)
            posterior_wishart_process = wishart_model['wishart process']
        else:
            x = np.tile(x, (D, 1)).T
            datagwp = (x, y)
            elbo = training_util.run_adam(posterior_wishart_process,datagwp,iterations=5000,minibatch_size=batch_size)
        a, b  = posterior_wishart_process.predict_mc(xtest, S)
        gwp_cov_pred[i,:,it*5:it*5+5] , gwp_mu_pred[i,:,it*5:it*5+5] = posterior_wishart_process.predict_mc(xtest, S)
        _, _, gwp_y_pred[i] = runpcg.sample_y( gwp_mu_pred[i],gwp_cov_pred[i] )

    #compute the total cov prediction (given all the data)
    gwp_sigma_list[i], _ = posterior_wishart_process.predict_mc(np.tile(X,(D,1)).T, S)

    #compute log likelihood
    loglik_gwp_list[i] = metrics.log_lik(Y[-N_test:], gwp_cov_pred[i], gwp_mu_pred[i])

    # use the conditional mu and sigma to sample data points

    # output prediction mean squared error
    ymse_gwp_list[i] = metrics.MSE(data[-N_test:], gwp_y_pred[i])

    #save covariance plot
    plot_pcg.plot_cov_comparison(X, Sigmas, gwp_sigma_list[i], D, "GWP", pred = gwp_cov_pred[i], save=f"gwp_simulation/gwp{i}", lim=(-4, 6))

plot_pcg.plot_cov_comparison(X, Sigmas, np.reshape(gwp_sigma_list, (n_sim * S, N, D, D)), D, "GWP", pred = np.reshape(gwp_cov_pred, (n_sim * S, N_test, D, D)), save=f"gwp_simulation/gwp{sim_type} average", lim = (-4, 6.5))

#save metrics for cov-cov
np.save(f"gwp_simulation/sigmas_{sim_type}", gwp_sigma_list)
np.save(f"gwp_simulation/mus_{sim_type}", gwp_mu_pred)
np.save(f"gwp_simulation/ysamples_{sim_type}", gwp_y_pred)

# sigma prediciton mean squared error
mse_gwp = metrics.MSE(Sigmas[-N_test:], np.reshape(gwp_cov_pred, (n_sim * S, N_test, D, D)))
np.save(f"gwp_simulation/mse_simulation_{sim_type}_cov", mse_gwp)

#save metrics for y-y
np.save(f"gwp_simulation/ymse_simulation_{sim_type}_cov", ymse_gwp_list)

correlations_loglik = np.ones((n_sim, S))
mse_resh =np.reshape(gwp_cov_pred, (n_sim , S, N_test, D, D))
for i in range(n_sim ):
    avgmse = np.mean(np.mean(mse_resh, axis = -1), axis = -1)
    ll = loglik_gwp_list[i]
    for s in range(S):
        correlations_loglik[i,s],_ = metrics.corr_timeseries(avgmse[i,s],ll)

correlations_loglik = np.reshape(correlations_loglik, (n_sim*S))
plt.hist(correlations_loglik, bins =20)
plt.title("Correlation between Sigma MSE and the log-likelihood")
plt.xlabel("correlation")
plt.ylabel("count")
plt.savefig(f"gwp_simulation/{sim_type}likelihoodcorr")
plt.close()
#sigma prediction for the test data
#averaged over the output dimensions
# mse = np.mean(np.reshape(np.triu(mse_mgarch[:, -N_test:]), (n_sim, N_test, D * D)), axis=-1)
# ymse = np.mean(ymse_mgarch_list, axis=-1)
correlations = np.zeros((n_sim, S, D))
pvalues = np.zeros_like(correlations)
mse_gwp_reshaped = np.reshape(mse_gwp, (n_sim, S , N_test , D , D))
for i in range(n_sim):
    for j in range(S):
        for d in range(D):
            ah = metrics.corr_timeseries(np.mean(mse_gwp_reshaped[i, j, :, d], axis = -1), ymse_gwp_list[i, j, :, d])
            correlations[i,j,d], pvalues[i,j,d] = ah

#plot the distribution of the correlations
#based on all samples collected
c = np.reshape(correlations, (n_sim * S * D))
plt.hist(c, bins =20)
plt.title("Correlation between Sigma MSE and Y MSE")
plt.xlabel("correlation")
plt.ylabel("count")
plt.savefig(f"gwp_simulation/{sim_type}msecorr")
plt.close()
#plot the distribution of the correlations
#per dimension
c = np.reshape(correlations, (n_sim * S, D))
for d in range(D):
    plt.hist(c[:,d], bins =20)
    plt.title(f"gwp_simulation/Correlation histogram of dimension {d}")
    plt.show()
plt.close()

#plot significance and correlation for different simulation data
indices = np.argsort(np.reshape(pvalues, (n_sim * S * D)))
c = np.reshape(correlations, (n_sim * S * D))
z = np.take(c, indices)
plt.scatter(np.take(pvalues, indices),z)
plt.title("correlation against significance")
plt.ylabel("correlation")
plt.xlabel("p-value")
plt.show()
##### correlation check ######

#compute the correlation for sigma-sigma and y-prediction y
ycor = np.zeros((n_sim, S, D))
sigmacor = np.zeros((n_sim,D,D))
for i in range(n_sim):
    data = np.load(f"mgarch_simulation/data{i}_{sim_type}.npy")
    for d in range(D):
        for sim in range(S):
            ycor[i,sim,d],_ = metrics.corr_timeseries(data[-N_test:,d], gwp_y_pred[i, sim, :, d])
        for e in range(D):
            sigmacor[i,d,e],_ =metrics.corr_timeseries(Sigmas[-N_test:,d,e], np.mean(gwp_cov_pred, axis = 1)[i, :,d, e])

#compute correlations between correlations
correlations_b = np.zeros((n_sim, S, d))
pvalues_b = np.zeros((n_sim, S, d))
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

print("output prediction metrics linear:")
print("average likelihood y|sigma, mu:", np.mean(loglik_gwp_list))
print("average mse y vs ypred", np.mean(ymse_gwp_list))
print("average correlation y with ypred:",np.mean(ycor))

print("sigma vs true sigma metrics")
print("average correlation sigma with sigma pred:",np.mean(sigmacor))
print("average mse sigma with sigma pred:", np.mean(mse_gwp))

print("mean correlation log likelihood and mse sigma",np.mean(correlations_loglik))
print("mean sigma mse vs y mse corr", np.mean(correlations))



print("end")
print("security")

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
N_test = 40 # number of test data 40
#retrain every 5 points
D = 3 # number of output dimensions
n_sim = 5# number of data simulations 8
S =150 #number of y samples given mu and sigma
sim_type = "linear_changing"
gwp_sigma_list = np.zeros((n_sim, S, N, D, D))
gwp_mu_pred = np.zeros((n_sim, S, N_test, D))

gwp_y_pred = np.zeros((n_sim, S, N_test, D))
gwp_cov_pred = np.zeros((n_sim, S, N_test, D, D))
ymse_gwp_list =np.zeros((n_sim, S, N_test, D))

correlations = []
loglik_gwp_list = np.zeros((n_sim, N_test))
num_iter = 65000
batch_size = 100
indu_rate = 0.2

for i in range(n_sim):
    #create dataset
    data = np.load(f"mgarch_simulation/data{i}_{sim_type}.npy")
    Sigmas = np.load(f"mgarch_simulation/trueSigma{i}_{sim_type}.npy")
    X = np.linspace(0, duration, N)
    Y = data

    #train model
    #train GWP
    for it,t in enumerate(range(N-N_test,N,5)):
        print(t)
        x = X[:t]
        y = Y[:t]
        ytest = Y[t:t+5]
        xtest = np.tile(X[t:t+5], (D, 1)).T
        if t == N-N_test:
            wishart_model = runpcg.run_BANNER(data=(x, y), T=duration, mnu = "independent",l_scale = 1.0, iterations=num_iter,num_inducing=int(indu_rate*N),batch_size=batch_size)
            posterior_wishart_process = wishart_model['wishart process']
        else:
            x = np.tile(x, (D, 1)).T
            datagwp = (x, y)
            elbo = training_util.run_adam(posterior_wishart_process,datagwp,iterations=5000,minibatch_size=batch_size)
        a, b  = posterior_wishart_process.predict_mc(xtest, S)
        gwp_cov_pred[i,:,it*5:it*5+5] , gwp_mu_pred[i,:,it*5:it*5+5] = posterior_wishart_process.predict_mc(xtest, S)
        _, _, gwp_y_pred[i] = runpcg.sample_y( gwp_mu_pred[i],gwp_cov_pred[i] )

    #compute the total cov prediction (given all the data)
    gwp_sigma_list[i], _ = posterior_wishart_process.predict_mc(np.tile(X,(D,1)).T, S)

    #compute log likelihood
    loglik_gwp_list[i] = metrics.log_lik(Y[-N_test:], gwp_cov_pred[i], gwp_mu_pred[i])

    # use the conditional mu and sigma to sample data points

    # output prediction mean squared error
    ymse_gwp_list[i] = metrics.MSE(data[-N_test:], gwp_y_pred[i])

    #save covariance plot
    plot_pcg.plot_cov_comparison(X, Sigmas, gwp_sigma_list[i], D, "GWP", pred = gwp_cov_pred[i], save=f"gwp_simulation/gwp{i}", lim=(-4, 6))

plot_pcg.plot_cov_comparison(X, Sigmas, np.reshape(gwp_sigma_list, (n_sim * S, N, D, D)), D, "GWP", pred = np.reshape(gwp_cov_pred, (n_sim * S, N_test, D, D)), save=f"gwp_simulation/gwp{sim_type} average", lim = (-4, 6.5))

#save metrics for cov-cov
np.save(f"gwp_simulation/sigmas_{sim_type}", gwp_sigma_list)
np.save(f"gwp_simulation/mus_{sim_type}", gwp_mu_pred)
np.save(f"gwp_simulation/ysamples_{sim_type}", gwp_y_pred)

# sigma prediciton mean squared error
mse_gwp = metrics.MSE(Sigmas[-N_test:], np.reshape(gwp_cov_pred, (n_sim * S, N_test, D, D)))
np.save(f"gwp_simulation/mse_simulation_{sim_type}_cov", mse_gwp)

#save metrics for y-y
np.save(f"gwp_simulation/ymse_simulation_{sim_type}_cov", ymse_gwp_list)

correlations_loglik = np.ones((n_sim, S))
mse_resh =np.reshape(gwp_cov_pred, (n_sim , S, N_test, D, D))
for i in range(n_sim ):
    avgmse = np.mean(np.mean(mse_resh, axis = -1), axis = -1)
    ll = loglik_gwp_list[i]
    for s in range(S):
        correlations_loglik[i,s],_ = metrics.corr_timeseries(avgmse[i,s],ll)

correlations_loglik = np.reshape(correlations_loglik, (n_sim*S))
plt.hist(correlations_loglik, bins =20)
plt.title("Correlation between Sigma MSE and the log-likelihood")
plt.xlabel("correlation")
plt.ylabel("count")
plt.savefig(f"gwp_simulation/{sim_type}likelihoodcorr")
plt.close()
#sigma prediction for the test data
#averaged over the output dimensions
# mse = np.mean(np.reshape(np.triu(mse_mgarch[:, -N_test:]), (n_sim, N_test, D * D)), axis=-1)
# ymse = np.mean(ymse_mgarch_list, axis=-1)
correlations = np.zeros((n_sim, S, D))
pvalues = np.zeros_like(correlations)
mse_gwp_reshaped = np.reshape(mse_gwp, (n_sim, S , N_test , D , D))
for i in range(n_sim):
    for j in range(S):
        for d in range(D):
            ah = metrics.corr_timeseries(np.mean(mse_gwp_reshaped[i, j, :, d], axis = -1), ymse_gwp_list[i, j, :, d])
            correlations[i,j,d], pvalues[i,j,d] = ah

#plot the distribution of the correlations
#based on all samples collected
c = np.reshape(correlations, (n_sim * S * D))
plt.hist(c, bins =20)
plt.title("Correlation between Sigma MSE and Y MSE")
plt.xlabel("correlation")
plt.ylabel("count")
plt.savefig(f"gwp_simulation/{sim_type}msecorr")
plt.close()
#plot the distribution of the correlations
#per dimension
c = np.reshape(correlations, (n_sim * S, D))
for d in range(D):
    plt.hist(c[:,d], bins =20)
    plt.title(f"gwp_simulation/Correlation histogram of dimension {d}")
    plt.show()
plt.close()

#plot significance and correlation for different simulation data
indices = np.argsort(np.reshape(pvalues, (n_sim * S * D)))
c = np.reshape(correlations, (n_sim * S * D))
z = np.take(c, indices)
plt.scatter(np.take(pvalues, indices),z)
plt.title("correlation against significance")
plt.ylabel("correlation")
plt.xlabel("p-value")
plt.show()
##### correlation check ######

#compute the correlation for sigma-sigma and y-prediction y
ycor = np.zeros((n_sim, S, D))
sigmacor = np.zeros((n_sim,D,D))
for i in range(n_sim):
    data = np.load(f"mgarch_simulation/data{i}_{sim_type}.npy")
    for d in range(D):
        for sim in range(S):
            ycor[i,sim,d],_ = metrics.corr_timeseries(data[-N_test:,d], gwp_y_pred[i, sim, :, d])
        for e in range(D):
            sigmacor[i,d,e],_ =metrics.corr_timeseries(Sigmas[-N_test:,d,e], np.mean(gwp_cov_pred, axis = 1)[i, :,d, e])

#compute correlations between correlations
correlations_b = np.zeros((n_sim, S, d))
pvalues_b = np.zeros((n_sim, S, d))
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

print("output prediction metrics linear_changing:")
print("average likelihood y|sigma, mu:", np.mean(loglik_gwp_list))
print("average mse y vs ypred", np.mean(ymse_gwp_list))
print("average correlation y with ypred:",np.mean(ycor))

print("sigma vs true sigma metrics")
print("average correlation sigma with sigma pred:",np.mean(sigmacor))
print("average mse sigma with sigma pred:", np.mean(mse_gwp))

print("mean correlation log likelihood and mse sigma",np.mean(correlations_loglik))
print("mean sigma mse vs y mse corr", np.mean(correlations))



print("end")
print("security")