import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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

def plot_sigma_ground_truth(x, Sigma):
    N, D, _ = Sigma.shape
    c1 = '#363537'
    c2 = '#EF2D56'
    fig, axes = plt.subplots(nrows=D, ncols=D, sharex=True, sharey=True, figsize=(10, 10))

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j < i:
                ax.axis('off')
            else:
                ax.plot(x, Sigma[:, i, j], color=c1, lw=2, label='Ground truth')
                ax.set_xlim([x[0], x[-1]])
                ax.set_title('$\sigma_{{{:d},{:d}}}(t)$'.format(i + 1, j + 1), fontsize=20)
    return fig, axes

def plot_cov_approximation(Y,X,Sigmas):
    fig, axes = plot_sigma_ground_truth(X, Sigmas)
    D = Y.shape[1]
    c1 = '#363537'
    c2 = '#EF2D56'

    cov_estimate = np.array([np.outer(y,y.T) for iy, y in enumerate(Y)])

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j < i:
                ax.axis('off')
            else:
                ax.scatter(X, cov_estimate[:,i,j], s =0.25,c=c2, label='estimated covariance from Y@Y.T')
                ax.set_xlim([X[0], X[-1]])
    return fig, axes


def plot_loss(num_iter, loss):
    x = np.linspace(1, num_iter, len(loss))
    fig = plt.figure()
    plt.plot(x, loss, label='ELBO')
    plt.xlim([x[0], x[-1]])
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.title('Training convergence')
    plt.tight_layout()
    plt.show()

#
def plot_sigma_predictions(samples, x, D, model,axes=None, plot_individuals=0, lim = None):
    posterior_expectation = tf.reduce_mean(samples, axis=0).numpy()
    posterior_variance = tf.math.reduce_variance(samples, axis=0).numpy()

    c1 = '#363537'
    c2 = '#EF2D56'

    if axes is None:
        print("axes was none")
        fig, axes = plt.subplots(nrows=D, ncols=D, sharex=True, sharey=True, figsize=(10, 10))

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j < i:
                ax.axis('off')
            else:
                mean = posterior_expectation[:, i, j]
                intv = 1.96*np.sqrt(posterior_variance[:, i, j])
                ax.plot(x, mean, lw=2, c=c2, label=f'{model} posterior mean')
                ax.fill_between(x, mean - intv, mean + intv, color=c2, alpha=0.2, label=f'{model} 95\% HDI')
                if plot_individuals > 0:
                    ixs = np.random.randint(0, samples.shape[0], size=plot_individuals)
                    for ix in ixs:
                        ax.plot(x, samples[ix, :, i, j].numpy(), c=c2, alpha=0.4, lw=0.5)
                ax.set_xlim([x[0], x[-1]])
                if lim is not None:
                    bottom, top = lim
                    ax.set_ylim(bottom, top)

def plot_cov_comparison(X, true_sigmas, sigma_samples,D,model = "BANNER", figure =None, save=None, lim = None):
    if figure is None:
        fig, axes = plot_sigma_ground_truth(X, true_sigmas)
    plot_sigma_predictions(sigma_samples, X, D,model, axes, lim = lim)

    axes[-1, -1].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('(Co)variance')
    handles, labels = axes[-1, -1].get_legend_handles_labels()
    plt.figlegend(handles, labels,
                  ncol=1, loc='lower left', bbox_to_anchor=(0.1, 0.10), frameon=False)
    plt.tight_layout()
    if save is None:
        plt.show()
    elif save:
        plt.savefig(save)
        plt.close(fig)
