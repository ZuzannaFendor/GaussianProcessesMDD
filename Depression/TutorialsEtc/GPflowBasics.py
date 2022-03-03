import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities import print_summary

plt.rcParams["figure.figsize"] = (12, 6)
print("This is a tutorial ")
X = np.array([0.1,0.2,0.4,0.45,1.1,1.15,1.6,2.6,2.9,3.1]).reshape(-1, 1)
Y = np.array([1.2,1.3,1.4,1.3,1.9,1.0 , 0.5,2.4,2.6,2.7]).reshape(-1, 1)

plt.plot(X, Y, "kx", mew=2)
plt.show()

k = gpflow.kernels.Matern52() + gpflow.kernels.Cosine(variance=0.8, lengthscales=2.0)

#mean_function = gpflow.mean_functions.Linear()
m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
print_summary(m)
m.likelihood.variance.assign(0.01)
m.kernel.kernels[0].lengthscales.assign(0.3)
print("vars",m.trainable_variables)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))



X_pred = np.linspace(0,20,200).reshape(200,1)
mean, var = m.predict_f(X_pred)
samples = m.predict_f_samples(X_pred, 10)

plt.figure(figsize=(12, 6))
plt.plot(X, Y, "kx", mew=2)
plt.plot(X_pred, mean, "C0", lw=2)
plt.fill_between(
    X_pred[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)

plt.plot(X_pred, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
_ = plt.xlim(-0.1, 20)
plt.show()

