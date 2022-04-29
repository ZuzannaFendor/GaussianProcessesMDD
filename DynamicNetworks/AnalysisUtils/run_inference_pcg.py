from Banner import src
import tensorflow as tf
import numpy as np
import gpflow
from gpflow.kernels import SquaredExponential, SharedIndependent, SeparateIndependent
from gpflow.inducing_variables import SharedIndependentInducingVariables, InducingPoints
from gpflow.ci_utils import ci_niter

from Banner.src.models.WishartProcess import WishartProcess
from Banner.util.training_util import run_adam
from Banner.src.kernels.PartlySharedIndependentMOK import PartlySharedIndependentMultiOutput
from Banner.src.likelihoods.WishartProcessLikelihood import WishartLikelihood

from rpy2.robjects import pandas2ri
import rpy2.robjects as ro

import numpy as np
import pandas as pd
from rpy2.robjects.conversion import localconverter


def run_BANNER(data, T, mnu = "shared", iterations=5000, num_inducing=None, learning_rate=0.01, batch_size=25):
    """
    :param data: Tuple (X, Y) of input and responses.
    :param T: Last timepoint (assume we start at t=0).
    :param mnu: the type of mu definition. (shared, independent or fully_dependent)
    :param iterations: Number of variational inference optimization iterations.
    :param num_inducing: Number of inducing points (inducing points are in the same space as X).
    :param learning_rate: Optimization parameter.
    :param batch_size: Data is split into batches of size batch_size, for stochastic optimization (otherwise we'll run
    out of memory).
    :return: Returns a dictionary with the posterior Wishart process (with trained hyperparameters and variational
    parameters), and a list with the ELBO (loss) per iteration.
    """
    print("running banner inference")

    X, Y = data
    N, D = Y.shape
    X = np.tile(X, (D, 1)).T
    data = (X,Y)
    if num_inducing is None:
        num_inducing = int(0.4 * N)

    model_inverse = False
    additive_noise = True

    nu = D + 1  # Degrees of freedom
    R = 10  # samples for variational expectation

    # in case of fully dependent mu,
    # the degrees of freedom for mu are exactly the same as for sigma
    # in all other cases, there is an additional degree of freedom used by
    # mu only
    if mnu == "fully_dependent":
        mnu_val = 0
    else:
        mnu_val = 1
    latent_dim = int((nu + mnu_val) * D)

    if num_inducing == N:
        Z_init = tf.identity(X)  # X.copy()
    else:
        Z_init = np.array(
            [np.linspace(0, T, num_inducing) for _ in range(D)]).T  # initial inducing variable locations
    Z = tf.identity(Z_init)
    iv = SharedIndependentInducingVariables(InducingPoints(Z))  # multi output inducing variables

    kernel_type = 'partially_shared'  # ['shared', 'separate', 'partially_shared']   # shares the same kernel parameters across input dimension
    kernel = SquaredExponential(lengthscales=5.)

    if kernel_type == 'shared':
        kernel = SharedIndependent(kernel, output_dim=latent_dim)
    elif kernel_type == 'separate':
        kernel = SeparateIndependent(
            [SquaredExponential(lengthscales=1. - (i + 6) * 0.01) for i in range(latent_dim)])
    elif kernel_type == 'partially_shared':
        kernel = PartlySharedIndependentMultiOutput(
            [SquaredExponential(lengthscales=0.5 + i * 0.5) for i in range(D)], nu=(nu + mnu_val))
    else:
        raise NotImplementedError

    # likelihood
    likelihood = WishartLikelihood(D, nu, mnu=mnu, R=R, additive_noise=additive_noise, model_inverse=model_inverse)
    # create GWP model

    wishart_process = WishartProcess(kernel, likelihood, D=D, nu=nu, inducing_variable=iv, mnu=mnu)

    # If num_inducing==N, we do not actually have inducing points.
    if num_inducing == N:
        gpflow.set_trainable(wishart_process.inducing_variable, False)

    elbo = run_adam(wishart_process, data, ci_niter(iterations), learning_rate, batch_size, natgrads=False, pb=True)
    return {'wishart process': wishart_process, 'ELBO': elbo}

def run_MOGP(data, iterations=5000, window_size = None , stride = 0):
    '''

    :param data: data: Tuple (X, Y) of input and responses.
    :param iterations: maximum number of iterations for training
    :param window_size: the size of the sliding window
    :return: a list of trained MOGP models.
    '''
    print("running MOGP inference")
    X, Y = data
    N, D = Y.shape
    #format the data to include the coregionalization label
    X, Y = __format_data(X,Y)
    # create the sliding window models
    if window_size is None:
        window_size = N
    models = __sliding_window(X, Y, D, window_size =window_size , stride=stride)
    #train the models
    for m in models:
        gpflow.optimizers.Scipy().minimize(
            m.training_loss, m.trainable_variables, options=dict(maxiter=iterations), method="L-BFGS-B",
        )
    return models

def __sliding_window(X, Y, D, window_size, stride):
    '''
    X: augmented X with (N*D)x2 where N is the number of samples, D is the dimnsionality and 2 are the "value" and
    "dimension label" columns
    Y: augmented Y with (N*D)x2 see above
    window_size: how many samples should be in one window
    stride: overlap between the windows
    returns: a list of trained models
    '''
    lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gaussian() for i in range(D)])
    output_dim = D  # Number of outputs
    rank = D  # Rank of W:  it is the number of degrees of correlation between the outputs.

    # Base kernel
    k = gpflow.kernels.Matern32(active_dims=[0])

    # Coregion kernel
    coreg = gpflow.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[1])

    kern = k * coreg  # changep

    nr_datapoints = X.shape[0]

    model_windows = []
    for i in range(0, nr_datapoints, (window_size - stride)):
        print(f"my {i}th window")
        if i + window_size < nr_datapoints:
            model_windows.append(
                gpflow.models.VGP((X[i:i + window_size], Y[i:i + window_size]), kernel=kern, likelihood=lik))
        else:
            model_windows.append(gpflow.models.VGP((X[i:-1], Y[i:-1]), kernel=kern, likelihood=lik))
    return model_windows

def __format_data(X,Y):
    '''

    :param X: the input (N,)
    :param Y: the output (N,D)
    :return: X (N*D,2),Y (N*D,2) formatted to include the coregionalization label,
    the first dimension might be actually smaller in reality when there are nan values present

    '''
    N,D = Y.shape
    X_aug = []
    Y_aug = []
    for n in range(N):
        for d in range(D):
            if not np.isnan(Y[n,d]) :
                Y_aug.append([Y[n,d], d])
                X_aug.append([X[n], d])
    return np.array(X_aug), np.array(Y_aug)

def run_MGARCH(data ):
    X,Y = data
    N, D = Y.shape
    print("running MGARCH inference")
    pd_rets = __MGARCH_data_preprocesssing(data)

    # compute DCC-Garch in R using rmgarch package
    pandas2ri.activate()
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_rets = ro.conversion.py2rpy(pd_rets)
    # convert the daily returns from pandas dataframe in Python to dataframe in R
    r_dccgarch_code = """
                    library('rmgarch')
                    function(r_rets, n_days){
                            univariate_spec <- ugarchspec(mean.model = list(armaOrder = c(0,0)),
                                                        variance.model = list(garchOrder = c(1,1),
                                                                            variance.targeting = FALSE, 
                                                                            model = "sGARCH"),
                                                        distribution.model = "norm")
                            n <- dim(r_rets)[2]
                            dcc_spec <- dccspec(uspec = multispec(replicate(n, univariate_spec)),
                                                dccOrder = c(1,1),
                                                distribution = "mvnorm")
                            gogarch_spec <-gogarchspec(mean.model = list(model = 'VAR', lag = 2), distribution.model = 'mvnorm', ica = 'fastica')

                            dcc_fit <- dccfit(dcc_spec, data=r_rets)
                            go_fit <- gogarchfit(gogarch_spec, data = r_rets, solver = 'hybrid', gfun = 'tanh', maxiter1 = 40000, epsilon = 1e-08, rseed = 100)

                            forecasts <- gogarchforecast(go_fit, n.ahead = n_days)
                            covariances = rcov(go_fit)
                            var = go_fit@model[["varcoef"]]
                            cof = coef(go_fit)
                            list(go_fit, forecasts@mforecast$H, covariances,cof )
                    }
                    """
    r_dccgarch = ro.r(r_dccgarch_code)
    n_days = 10
    r_res = r_dccgarch(r_rets, n_days)

    pandas2ri.deactivate()
    # end of R

    r_gogarch_model = r_res[0]  # model parameters

    r_forecast_cov = r_res[1]  # forecasted covariance matrices for n_days

    r_cov = r_res[2]  # the covarince matrices for known points
    coef = r_res[3]


    mgarch_sigma = np.zeros((N, D, D))
    for i in range(N * D * D):
        mgarch_sigma[i // (D * D), (i % (D * D)) // D, (i % (D * D)) % D] = r_cov[i]


    return {'model':r_gogarch_model ,'forecast':r_forecast_cov,'covariance_matrix':mgarch_sigma, "coefficients":coef}

def __MGARCH_data_preprocesssing(data):
    X,Y = data
    df = pd.DataFrame(Y, columns=['Column_A', 'Column_B', 'Column_C'])
    pd_rets = pd.DataFrame(df)
    return pd_rets
