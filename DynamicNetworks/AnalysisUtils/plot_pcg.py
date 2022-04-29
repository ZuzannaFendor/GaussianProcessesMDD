import matplotlib.pyplot as plt
import numpy as np

def plot_gp(x, mu, var, color, label):
    plt.plot(x, mu, color=color, lw=2, label=label)
    plt.fill_between(
        x[:, 0],
        (mu - 2 * np.sqrt(var))[:, 0],
        (mu + 2 * np.sqrt(var))[:, 0],
        color=color,
        alpha=0.4,
    )

def plot_timeseries(Xtest, mus, vs, X, Y):
    plt.figure(figsize=(20, 10))
    for i, (mu, var) in enumerate(zip(mus, vs)):
        (line,) = plt.plot(X, Y[:,i], "x", mew=2)
        label = "output " + str(i)
        mu = mu[:,None]
        var = var[:,None]
        plot_gp( Xtest, mu, var ,line.get_color(), label)
    plt.legend()
    plt.show()

