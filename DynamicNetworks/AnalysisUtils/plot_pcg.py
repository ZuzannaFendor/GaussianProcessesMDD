import matplotlib.pyplot as plt
import numpy as np

def plot_gp(x, mu, var, color, label):
    plt.plot(x, mu, color=color, lw=2, label=label)
    plt.fill_between(
        x,
        (mu - 2 * np.sqrt(var))[:, 0],
        (mu + 2 * np.sqrt(var))[:, 0],
        color=color,
        alpha=0.4,
    )

def plot_timeseries(Xtest, mus, vs, X, Y):
    print(f"Xtest{Xtest.shape}, mus {mus.shape} vs {vs.shape}, X {X.shape} Y {Y.shape}")
    plt.figure(figsize=(20, 10))
    for i, (mu, var) in enumerate(zip(mus, vs)):
        (line,) = plt.plot(X, Y[:,i], "x", mew=2)
        label = "output " + str(i)
        mu = mu[:,None]
        var = var[:,None]
        plot_gp( Xtest, mu, var ,line.get_color(), label)
    plt.legend()
    plt.show()

def plot_covariance_matrix(Sigma):
    Sigma_mean= np.mean(Sigma, axis = 0)
    _ = plt.imshow(Sigma_mean)
    plt.show()

def plot_mse(mses, var_labels, modelname ):
    '''
    Plots the box plots of the mse analysis of a model
    - the box plot showing separate plots per variable
    - the box plot showing
    :param mses: (R x N x D) array of the computed mean squared errors

    '''
    D = mses.shape[-1]
    positions = list(range(1,D+1))

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle(f"Mean Squared Errors of {modelname}")
    # Creating axes instance

    # Creating plot
    bp = ax.boxplot(np.reshape(mses , (-1,D)))
    plt.xticks(positions, var_labels,
               rotation=10)

    # show plot
    plt.show()


    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle(f"Mean Squared Error of {modelname} averaged over all variables")
    # Creating axes instance
    # Creating plot
    bp = ax.boxplot(mses.flatten())
    # show plot
    plt.show()
