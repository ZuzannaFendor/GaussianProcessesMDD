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
import gpflow
from Banner.util import training_util

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def plot_gp(x, mu, var, color, label):
    plt.plot(x, mu, color=color, lw=2, label=label)
    plt.fill_between(
        x[:, 0],
        (mu - 2 * np.sqrt(var))[:, 0],
        (mu + 2 * np.sqrt(var))[:, 0],
        color=color,
        alpha=0.4,
    )


def plot(m):
    plt.figure(figsize=(8, 4))
    Xtest = np.linspace(20, 300, 100)[:, None]
    (line,) = plt.plot(X1, Y1, "x", mew=2)
    mu, var = m.predict_f(np.hstack((Xtest, np.zeros_like(Xtest))))
    plot_gp(Xtest, mu, var, line.get_color(), "Y1")

    (line,) = plt.plot(X2, Y2, "x", mew=2)
    mu, var = m.predict_f(np.hstack((Xtest, np.ones_like(Xtest))))
    plot_gp(Xtest, mu, var, line.get_color(), "Y2")

    plt.legend()
    plt.show()

data, colnames, scl = datapcg.import_ESMdata()
X = data['hour_no'].to_numpy(dtype="float64")[:400]
Y = data.loc[:,['neg_affect','pos_affect','sus','worry','mental_unrest']].to_numpy(dtype="float64")[:400,:]
D = 5

Tmax  = np.max(X)
Tmin = np.min(X)
augX, augY = runpcg.format_data(X,Y)
N = augX.shape[0]
output_dim = D  # Number of outputs
rank = 1 # Rank of W
k2 = gpflow.kernels.Matern32(active_dims=[0])

# Coregion kernel
coreg2 = gpflow.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[1])

kern2 = k2* coreg2

lik2 = gpflow.likelihoods.SwitchedLikelihood(
    [gpflow.likelihoods.Gaussian() for i in range(D)]
)


# Base kernel
k = gpflow.kernels.Matern32(active_dims=[0])

# Coregion kernel
coreg = gpflow.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[1])

kern = k * coreg


# now build the GP model as normal
M = int(N*0.1)


#regular inducing points

Z, _ = datapcg.stackify_data(np.linspace(Tmin,Tmax,M),np.ones((M,D)))
Z = np.array(Z)
lik = gpflow.likelihoods.SwitchedLikelihood(
    [gpflow.likelihoods.Gaussian() for i in range(D)]
)

m2 = gpflow.models.VGP((augX,augY), kernel=kern2, likelihood=lik2)

m = gpflow.models.SVGP( kernel=kern, likelihood=lik,inducing_variable=Z,num_data=N)


#train models
# fit the covariance function parameters
maxiter = 500
gpflow.optimizers.Scipy().minimize(
    m2.training_loss,
    m2.trainable_variables,
    options=dict(maxiter=maxiter),
    method="L-BFGS-B",
)

gpflow.set_trainable(m.inducing_variable, False)

post = training_util.run_adam(m,(augX, augY),iterations = maxiter,learning_rate=0.1, minibatch_size=20, natgrads=True, pb=True)

X1 = np.random.rand(200, 1)  # Observed locations for first output
X2 = np.random.rand(100, 1) * 0.5  # Observed locations for second output

Y1 = np.sin(6 * X1) + np.random.randn(*X1.shape) * 0.03
Y2 = np.sin(6 * X2 + 0.7) + np.random.randn(*X2.shape) * 0.1

plt.figure(figsize=(8, 4))
plt.plot(X1, Y1, "x", mew=2)
_ = plt.plot(X2, Y2, "x", mew=2)
X_augmented = np.vstack((np.hstack((X1, np.zeros_like(X1))), np.hstack((X2, np.ones_like(X2)))))

# Augment the Y data with ones or zeros that specify a likelihood from the list of likelihoods
Y_augmented = np.vstack((np.hstack((Y1, np.zeros_like(Y1))), np.hstack((Y2, np.ones_like(Y2)))))

# This likelihood switches between Gaussian noise with different variances for each f_i:
lik = gpflow.likelihoods.SwitchedLikelihood(
    [gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Gaussian()]
)
output_dim = 2  # Number of outputs
rank = 1  # Rank of W
k2 = gpflow.kernels.Matern32(active_dims=[0])

# Coregion kernel
coreg2 = gpflow.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[1])

kern2 = k2* coreg2


lik2 = gpflow.likelihoods.SwitchedLikelihood(
    [gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Gaussian()]
)

# now build the GP model as normal
m2 = gpflow.models.VGP((X_augmented, Y_augmented), kernel=kern2, likelihood=lik2)

# fit the covariance function parameters
maxiter = 500
gpflow.optimizers.Scipy().minimize(
    m2.training_loss,
    m2.trainable_variables,
    options=dict(maxiter=maxiter),
    method="L-BFGS-B",
)

# Base kernel
k = gpflow.kernels.Matern32(active_dims=[0])

# Coregion kernel
coreg = gpflow.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[1])

kern = k * coreg


# now build the GP model as normal

N =X_augmented.shape[0]
M = int(N*0.8)
M1 = int(200*0.1)
M2 = int(100*0.1)
# np.random.shuffle(X_augmented)
Z = X_augmented[:M, :].copy()
#regular inducing points
Zinit = np.linspace(0, 1, M)[:, None]
stackedM1= np.hstack((np.linspace(0,1,M1)[:,None], np.zeros((M1,1))))
stackedM2 = np.hstack((np.linspace(0,1,M2)[:,None], np.ones((M2,1))))
Z = np.vstack((stackedM1,stackedM2))
m = gpflow.models.SVGP( kernel=kern, likelihood=lik,inducing_variable=Z,num_data=N)

gpflow.set_trainable(m.inducing_variable, False)

post = training_util.run_adam(m,(X_augmented,Y_augmented),iterations = 5000,learning_rate=0.1, minibatch_size=20, natgrads=True, pb=True)









def plot_gp(x, mu, var, color, label):
    plt.plot(x, mu, color=color, lw=2, label=label)
    plt.fill_between(
        x[:, 0],
        (mu - 2 * np.sqrt(var))[:, 0],
        (mu + 2 * np.sqrt(var))[:, 0],
        color=color,
        alpha=0.4,
    )


def plot(m):
    plt.figure(figsize=(8, 4))
    Xtest = np.linspace(0, 1, 100)[:, None]
    (line,) = plt.plot(X1, Y1, "x", mew=2)
    mu, var = m.predict_f(np.hstack((Xtest, np.zeros_like(Xtest))))
    plot_gp(Xtest, mu, var, line.get_color(), "Y1")

    (line,) = plt.plot(X2, Y2, "x", mew=2)
    mu, var = m.predict_f(np.hstack((Xtest, np.ones_like(Xtest))))
    plot_gp(Xtest, mu, var, line.get_color(), "Y2")

    plt.legend()
    plt.show()


plot(m)

B = coreg.output_covariance().numpy()
print("B =", B)
_ = plt.imshow(B)
plt.show()

print("finished")






def generate_data(N=100):
    X = np.random.rand(N)[:, None] * 10 - 5  # Inputs = N x D
    G = np.hstack((0.5 * np.sin(3 * X) + X, 3.0 * np.cos(X) - X))  # G = N x L
    W = np.array([[0.5, -0.3, 1.5], [-0.4, 0.43, 0.0]])  # L x P
    F = np.matmul(G, W)  # N x P
    Y = F + np.random.randn(*F.shape) * [0.2, 0.2, 0.2]

    return X, Y
Xgen,Ygen = datagen = generate_data()


print(Xgen,Ygen)






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