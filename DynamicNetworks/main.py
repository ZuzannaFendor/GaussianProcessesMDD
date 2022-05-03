#imports

import Banner
import AnalysisUtils as AU
from AnalysisUtils import run_inference_pcg as runpcg
import numpy as np
import matplotlib.pyplot as plt
from AnalysisUtils import plot_pcg as plotpcg
from AnalysisUtils import data_processing_pcg as dataprpcg

from AnalysisUtils import run_inference_pcg as infer
D, N, T = 3, 100, 4.0

X = np.linspace(0,T,N) # np.tile(np.linspace(0,T,N), (D, 1)).T
Y = np.array([np.linspace(0,T,N)+np.random.rand(N),np.ones(N)+np.random.rand(N)*0.2,np.linspace(T,0,N)*np.random.rand(N)])
Y = Y.T
print(Y.shape)

# plt.scatter(X, Y[:,0])
# plt.scatter(X, Y[:,1])
# plt.scatter(X, Y[:,2])
# plt.show()

num_iter = 50
num_samples = 200
testpoints = np.tile(np.linspace(0, T, N), (D, 1)).T
print(testpoints.shape)

# wishart_model = runpcg.run_BANNER(data=(X, Y), mnu = "shared", T=T,iterations=num_iter,num_inducing=int(0.4*N),batch_size=100)
# posterior_wishart_process = wishart_model['wishart process']
# sigma_samples_gwp , mu_samples_gwp= posterior_wishart_process.predict_mc(testpoints, num_samples)
# sigma_mean_gwp, mu_mean_gwp = posterior_wishart_process.predict_map(testpoints)

# y_samples = np.zeros_like(mu_samples_gwp)
#
# for s in range(num_samples):
#     for t in range(int(N)):
#         y_samples[s,t] = np.random.multivariate_normal(mu_samples_gwp[s,t], sigma_samples_gwp[s,t])
# mu_y = np.mean(y_samples,axis = 0)
# var_y = np.var(y_samples, axis = 0 )
#
# plotpcg.plot_timeseries(testpoints, mu_y.T, var_y.T, X, Y)

mogp_model = runpcg.run_MOGP(data=(X,Y))[0]
aug_test_X, _ = dataprpcg.stackify_data(testpoints[0],np.ones((N,D)))
print(np.array(aug_test_X))
mu_y,var_y = mogp_model.predict_f(np.array(aug_test_X))

plotpcg.plot_timeseries(testpoints, mu_y.T, var_y.T, X, Y)


# mgarch_model= runpcg.run_MGARCH(data=(X,Y))
# mgarch_covariance = mgarch_model["covariance_matrix"]
# print("finished", mgarch_covariance)

