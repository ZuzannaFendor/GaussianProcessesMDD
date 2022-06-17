#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import Banner
import AnalysisUtils as AU
from AnalysisUtils import run_inference_pcg as runpcg
from AnalysisUtils import plot_pcg as plotpcg
from AnalysisUtils import data_processing_pcg as dataprpcg
from AnalysisUtils import metrics
import time


#data characteristics
# D, N, T = 5, 200, 4.0

#generate data

# X = np.linspace(0,T,N) # np.tile(np.linspace(0,T,N), (D, 1)).T
# Y = np.array([np.sin(np.linspace(0,T,N))+np.random.rand(N),np.linspace(0,T,N)*0.3+np.random.rand(N),np.linspace(0,T,N)+np.random.rand(N),np.ones(N)+np.random.rand(N)*0.2,np.linspace(T,0,N)*np.random.rand(N)])
# Y = Y.T

data, colnames, scl = dataprpcg.import_ESMdata()
X = data['hour_no'].to_numpy(dtype="float64")[:30]
Y = data.loc[:,['neg_affect','pos_affect','sus','worry','mental_unrest']].to_numpy(dtype="float64")[:30,:]

datareg= dataprpcg.resample_data(data.loc[:,['neg_affect','pos_affect','sus','worry','mental_unrest']],"nearest","3H" )
regX = datareg['neg_affect'].to_numpy(dtype="float64")[2:]
regY = datareg.loc[:,['neg_affect','pos_affect','sus','worry','mental_unrest']].to_numpy(dtype="float64")[2:,:]

N, D, T = Y.shape[0],Y.shape[1], np.max(X)
# plot data in a scatterplot
# plt.scatter(X, Y[:,0])
# plt.scatter(X, Y[:,1])
# plt.scatter(X, Y[:,2])
# plt.show()


num_iter = 20
num_samples = 200
N_test = 60
testpoints = np.linspace(0, T, N_test)
tiled_testpoints = np.tile(testpoints, (D, 1)).T



#data = (X,Y) #(np.array([0,1,2,3,4,5,6,7,8])*2,np.array([[2],[5],[8],[4],[8],[10],[11],[20],[21]]))

# models = ["BANNER","MOGP","MGARCH"]
# var_names = ["sore throat","headache","fever","cough", "runny nose"]
# for m in models:
#
#     labels, predictions = runpcg.cross_validate(m,(X,Y), n_splits = 3, test_s=2, num_iter = 500, num_samples = 200 )
#     #use only with mgarch!
#     if m == "MGARCH":
#         predictions = np.reshape(predictions, (1,predictions.shape[0],predictions.shape[1]))
#     #use only with mogp
#     if m == "MOGP":
#         predictions = runpcg.reformat_data(predictions, D, samples = True)
#     mse = metrics.MSE(labels, predictions)
#     plotpcg.plot_mse(mse,var_names,m)
#     correlation = metrics.correlation(labels, predictions)


# LABELS, PREDICTIONS=runpcg.cross_validate("BANNER",(X,Y))
# #labels, predictions = np.random.random((4,40,5)), np.random.random(size = (4,200,40,5))
# labels, predictions = np.array(LABELS), np.array(PREDICTIONS)
# print(f"Lables{labels.shape} and predictions{predictions.shape} ")
# labels = np.reshape(labels, (4*40,5))
# predictions = np.reshape(predictions, (200,4*40,5))

# labels = np.random.random((160,5))
# predictions = np.random.random((40,160,5))
# print(f"shape after flattening Lables{labels.shape} and predictions{predictions.shape}")
# mse_mean, mse_var = metrics.MSE(labels,predictions)
# corr = metrics.correlation(labels, predictions)
# print("finished")
# compute mse distribution


#### Straight out running and fitting the models ####
#### using all of the data to provide us with a  ####
#### model of the depression progression         ####

#
tic = time.perf_counter()
wishart_model = runpcg.run_BANNER(data=(X, Y), mnu = "shared", T=T,iterations=num_iter,num_inducing=int(0.4*N),batch_size=100)
posterior_wishart_process = wishart_model['wishart process']
sigma_samples_gwp , mu_samples_gwp= posterior_wishart_process.predict_mc(tiled_testpoints, num_samples)
sigma_mean_gwp, mu_mean_gwp = posterior_wishart_process.predict_map(tiled_testpoints)
y_mu_gwp, y_var_gwp, y_samples = runpcg.sample_y(mu_samples_gwp,sigma_samples_gwp)
toc = time.perf_counter()
print(f"wishart took {toc-tic} ns to run in total")
plotpcg.plot_timeseries(testpoints, y_mu_gwp.T, y_var_gwp.T, X, Y)

tic = time.perf_counter()
mogp_model = runpcg.run_MOGP(data=(X,Y), iterations = 100)[0]
aug_test_X, _ = dataprpcg.stackify_data(testpoints,np.ones((N_test,D)))
mu_y,var_y = mogp_model.predict_f(np.array(aug_test_X))
mumu = np.reshape(mu_y,(N_test,D)).T
vuvu = np.reshape(var_y,(N_test,D)).T
toc = time.perf_counter()
print(f"mogp took {toc-tic} ns to run in total")
plotpcg.plot_timeseries(testpoints, mumu, vuvu, X, Y)
# Xtest(60,), mus (3, 60) vs (3, 60), X (1476,) Y (1476, 3)

isna = pd.DataFrame(regY).isnull().values.any()
isna = pd.DataFrame(regX).isnull().values.any()
mgarch_model= runpcg.run_MGARCH(data=(regX,regY))
mgarch_covariance = mgarch_model["covariance_matrix"]
ntrain, ntest = 100, 20
# forecast_mu, forecast_sigma = runpcg.forecast_MGARCH((regX,regY), ntrain, ntest )
print("finished")
