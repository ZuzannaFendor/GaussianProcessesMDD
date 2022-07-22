from Banner import src
import tensorflow as tf
import gpflow
from gpflow.kernels import SquaredExponential, SharedIndependent, SeparateIndependent
from gpflow.inducing_variables import SharedIndependentInducingVariables, InducingPoints
from gpflow.ci_utils import ci_niter
from gpflow.utilities import print_summary

from Banner.src.models.WishartProcess import WishartProcess
from Banner.util.training_util import run_adam
from Banner.src.kernels.PartlySharedIndependentMOK import PartlySharedIndependentMultiOutput
from Banner.src.likelihoods.WishartProcessLikelihood import WishartLikelihood

from rpy2.robjects import pandas2ri
import rpy2.robjects as ro

import numpy as np
import pandas as pd
from rpy2.robjects.conversion import localconverter
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit


def run_BANNER(data, T, mnu = "shared", l_scale =1., iterations=5000, num_inducing=None, learning_rate=0.01, batch_size=25):
    """
    :param data: Tuple (X, Y) of input and responses.
    :param T: Last timepoint of the time series (assume we start at t=0).
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
    if mnu =="shared" or mnu == "independent":
        mnu_val = 1
    else:
        mnu_val = 0
    latent_dim = int((nu + mnu_val) * D)

    if num_inducing == N:
        Z_init = tf.identity(X)  # X.copy()
    else:
        Z_init = np.array(
            [np.linspace(0, T, num_inducing) for _ in range(D)]).T  # initial inducing variable locations
    Z = tf.identity(Z_init)
    iv = SharedIndependentInducingVariables(InducingPoints(Z))  # multi output inducing variables

    kernel = construct_kernel(latent_dim, nu, mnu_val,D,l_scale)


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

def run_svgpMOGP(data,  iterations=5000, inducing_ratio=0.4, batch_size=100, window_size = None , stride = 0):
    X,Y = data
    N,D = Y.shape
    Tmax = np.max(X)
    Tmin = np.min(X)
    augX, augY = format_data(X, Y)
    Naug = augX.shape[0]
    rank = 1  # Rank of W

    #kernel
    k = gpflow.kernels.Matern32(active_dims=[0])

    # Coregion kernel
    coreg = gpflow.kernels.Coregion(output_dim=D, rank=rank, active_dims=[1])

    kern = k * coreg


    M = int(N * inducing_ratio)


    Z, _ = format_data(np.linspace(Tmin, Tmax, M), np.ones((M, D)))
    Z = np.array(Z)
    lik = gpflow.likelihoods.SwitchedLikelihood(
        [gpflow.likelihoods.Gaussian() for i in range(D)]
    )

    m = gpflow.models.SVGP(kernel=kern, likelihood=lik, inducing_variable=Z, num_data=Naug)

    # train models
    # fit the covariance function parameters
    gpflow.set_trainable(m.inducing_variable, False)

    elbo = run_adam(m, (augX, augY), iterations=iterations, learning_rate=0.1, minibatch_size=20,
                                  natgrads=True, pb=True)
    return {'gaussian process': m, 'ELBO': elbo}

def run_example(data, lower=-8., upper=8.):
    MAXITER = ci_niter(2000)

    X,Y = data
    N = Y.shape[0]  # number of points
    D = 1  # number of input dimensions
    M = 15  # number of inducing points

    P = Y.shape[1]  # number of observations = output dimensions
    L = P #2  # number of latent GPs
    def generate_data(N=100):
        X = np.random.rand(N)[:, None] * 10 - 5  # Inputs = N x D
        G = np.hstack((0.5 * np.sin(3 * X) + X, 3.0 * np.cos(X) - X))  # G = N x L
        W = np.array([[0.5, -0.3, 1.5], [-0.4, 0.43, 0.0]])  # L x P
        F = np.matmul(G, W)  # N x P
        Y = F + np.random.randn(*F.shape) * [0.2, 0.2, 0.2]

        return X, Y

    # X, Y = data = generate_data(N)


    Zinit = np.linspace(int(np.min(X)), int(np.max(X)), M)[:, None]

    def plot_model(m, lower=-8.0, upper=8.0):
        pX = np.linspace(lower, upper, 100)[:, None]
        pY, pYv = m.predict_y(pX)
        if pY.ndim == 3:
            pY = pY[:, 0, :]
        plt.plot(X, Y, "x")
        plt.gca().set_prop_cycle(None)
        plt.plot(pX, pY)
        for i in range(pY.shape[1]):
            top = pY[:, i] + 2.0 * pYv[:, i] ** 0.5
            bot = pY[:, i] - 2.0 * pYv[:, i] ** 0.5
            plt.fill_between(pX[:, 0], top, bot, alpha=0.3)
        plt.xlabel("X")
        plt.ylabel("f")
        plt.title(f"ELBO: {m.elbo(data):.3}")
        plt.plot(Z, Z * 0.0, "o")
        plt.show()

    # create multi-output kernel
    # kernel = gpflow.kernels.SharedIndependent(
    #     gpflow.kernels.SquaredExponential() , output_dim=P
    # )
    kern_list = [gpflow.kernels.SquaredExponential() + gpflow.kernels.Linear() for _ in range(L)]
    # Create multi-output kernel from kernel list
    kernel = gpflow.kernels.LinearCoregionalization(
        kern_list, W=np.random.randn(P, L)
    )
    # initialization of inducing input locations (M random points from the training inputs)
    Z = Zinit.copy()
    # create multi-output inducing variables from Z
    iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
        gpflow.inducing_variables.InducingPoints(Z)
    )

    # initialize mean of variational posterior to be of shape MxL
    q_mu = np.zeros((M, L))
    # initialize \sqrt(Σ) of variational posterior to be of shape LxMxM
    q_sqrt = np.repeat(np.eye(M)[None, ...], L, axis=0) * 1.0

    # create SVGP model as usual and optimize
    m = gpflow.models.SVGP(
        kernel, gpflow.likelihoods.Gaussian(), inducing_variable=iv, q_mu=q_mu, q_sqrt=q_sqrt
    )
    #
    m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=P)

    return m

    def optimize_model_with_scipy(model):
        optimizer = gpflow.optimizers.Scipy()
        optimizer.minimize(
            model.training_loss_closure(data),
            variables=model.trainable_variables,
            method="l-bfgs-b",
            options={"disp": True, "maxiter": MAXITER},
        )

    optimize_model_with_scipy(m)
    print_summary(m)

    plot_model(m, lower, upper)


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
    rank = 2 #D  # Rank of W:  it is the number of degrees of correlation between the outputs.

    # Base kernel
    k = gpflow.kernels.Matern32(active_dims=[0])

    # Coregion kernel
    coreg = gpflow.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[1])

    kern = k * coreg  # changep

    nr_datapoints = X.shape[0]

    model_windows = []
    for i in range(0, nr_datapoints, (window_size - stride)):
        if i + window_size < nr_datapoints:
            # format the data to include the coregionalization label
            x_window,y_window = format_data(X[i:i + window_size], Y[i:i + window_size])
            model_windows.append(
                gpflow.models.VGP((x_window,y_window), kernel=kern, likelihood=lik))
        else:
            x_window,y_window = format_data(X[i:-1], Y[i:-1])
            model_windows.append(gpflow.models.VGP((x_window,y_window), kernel=kern, likelihood=lik))
    return model_windows

def format_data(X, Y):
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

def reformat_data(X,D, samples = False):
    N = int(X.shape[-2]/D)

    if samples:
        nsampl = X.shape[0]
        deformated_X = np.reshape(X, (nsampl, N, D))
    else:
        deformated_X = np.reshape(X,(N, D))
    return deformated_X

# def forecast_MGARCH(data, ntrain,ntest ):
#     X, Y = data
#     N, D = Y.shape
#     print("running MGARCH rollout")
#     pd_rets = __MGARCH_data_preprocesssing(data)
#
#     # compute DCC-Garch in R using rmgarch package
#     pandas2ri.activate()
#     with localconverter(ro.default_converter + pandas2ri.converter):
#         r_rets = ro.conversion.py2rpy(pd_rets)
#     # convert the daily returns from pandas dataframe in Python to dataframe in R
#     r_dccgarch_code = """
#                         library('rmgarch')
#                         function(r_rets, npred, nout){
#                                 univariate_spec <- ugarchspec(mean.model = list(armaOrder = c(0,0)),
#                                                         variance.model = list(garchOrder = c(1,1),
#                                                                             variance.targeting = FALSE,
#                                                                             model = "sGARCH"),
#                                                         distribution.model = "norm")
#                                 n <- dim(r_rets)[2]
#
#                                 gogarch_spec <-gogarchspec(mean.model = list(model = 'VAR', lag = 1),distribution.model = 'mvnorm', ica = 'fastica')
#                                 gogarch_fit <- gogarchfit(gogarch_spec, data=r_rets,  out.sample = nout,gfun = "tanh")
#
#                                 forecast <- gogarchforecast(gogarch_fit, n.ahead = npred, n.roll = 0)
#
#                                 cov <- rcov(forecast)
#                                 mean <- fitted(forecast)
#
#                                 list(cov,mean)
#                         }
#                         """
#     r_dccgarch = ro.r(r_dccgarch_code)
#     nout = N - (ntrain+ntest)
#     npred = ntest
#     r_res = r_dccgarch(r_rets, npred, nout)
#
#     pandas2ri.deactivate()
#     # end of R
#
#     forecast_cov = r_res[0]  # model parameters
#     forecast_mean = r_res[1]
#     datafr = pd.DataFrame(forecast_cov)
#     datafrmean = pd.DataFrame(forecast_mean).T
#     mgarch_sigma = np.zeros((ntest, D, D))
#
#     for i in range(ntest * D * D):
#         mgarch_sigma[i // (D * D), (i % (D * D)) // D, (i % (D * D)) % D] = datafr[i]
#     mgarch_mean= np.zeros((ntest, D))
#     print(forecast_mean)
#     for i in range(ntest * D ):
#         col_D = i // ntest
#         row_n = i % ntest
#         dat = datafrmean[i]
#         mgarch_mean[row_n, col_D] = dat
#     return mgarch_mean, mgarch_sigma

def run_MGARCH(data, refit_every = 5, nrTest = 20):
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
                    library('rugarch')
                    library('rmgarch')
                    function(r_rets, refit_every, nrTest){
                            univariate_spec <- ugarchspec(mean.model = list(armaOrder = c(0,0)),
                                                        variance.model = list(garchOrder = c(1,1),
                                                                            variance.targeting = FALSE, 
                                                                            model = "sGARCH"),
                                                        distribution.model = "norm")
                            n <- dim(r_rets)[2]
                            lag = 2
                            
                            dcc_spec <- dccspec(uspec = multispec(replicate(n, univariate_spec)),VAR = TRUE,
                                                robust = FALSE, lag = lag,
                                                dccOrder = c(1,1),
                                                distribution = "mvnorm"
                                                )
                             
                            dcc_fit <- dccfit(dcc_spec, data=na.omit(r_rets))
                            
                            forecasts <- dccroll(dcc_spec,data=na.omit(r_rets), n.ahead = 1, forecast.length =nrTest, refit.every = refit_every)
                            covariances = rcov(dcc_fit)
                            var = dcc_fit@model[["varcoef"]]
                            cof = coef(dcc_fit)
                            means = fitted(forecasts)
                            pred_cov = rcov(forecasts)
                            list(dcc_fit, forecasts, covariances,cof ,means,pred_cov)
                    }
                    """
    r_dccgarch = ro.r(r_dccgarch_code)
    r_res = r_dccgarch(r_rets, refit_every, nrTest)

    pandas2ri.deactivate()
    # end of R

    r_model = r_res[0]  # model parameters

    r_forecast_cov = r_res[1]  # forecasted covariance matrices for n_days

    r_cov = r_res[2]  # the covarince matrices for known points
    coef = r_res[3]
    r_forecast_condmean =r_res[4]
    r_forecast_condcov = r_res[5]

    mgarch_sigma = np.zeros((N, D, D))
    for i in range(N * D * D):
        mgarch_sigma[i // (D * D), (i % (D * D)) // D, (i % (D * D)) % D] = r_cov[i]

    pred_mu = np.zeros((nrTest,D))
    for j in range(nrTest * D):
        n = j%nrTest
        d = j//nrTest
        pred_mu[n,d] = r_forecast_condmean[j]

    mgarch_sigma_pred = np.zeros((nrTest, D, D))
    for i in range(nrTest * D * D):
        mgarch_sigma_pred[i // (D * D), (i % (D * D)) // D, (i % (D * D)) % D] = r_forecast_condcov[i]

    return {'model':r_model ,'forecast_cov':r_forecast_cov,'covariance_matrix':mgarch_sigma, "coefficients":coef, "mu_predictions":pred_mu, "cov_predictions":mgarch_sigma_pred}

def construct_kernel( latent_dim, nu, mnu_val,D, l_scale):
    kernel_type = 'partially_shared'  # ['shared', 'separate', 'partially_shared']   # shares the same kernel parameters across input dimension

    # kernel = gpflow.kernels.Exponential(lengthscales=l_scale) * gpflow.kernels.Periodic(base_kernel=gpflow.kernels.SquaredExponential(lengthscales=l_scale))
    kernel = gpflow.kernels.Exponential(lengthscales=l_scale) #* gpflow.kernels.Periodic(base_kernel=gpflow.kernels.SquaredExponential(lengthscales=l_scale), period =3.)
    set_untrainable_variance(kernel)

    if kernel_type == 'shared':
        kernel = SharedIndependent(kernel, output_dim=latent_dim)
    elif kernel_type == 'separate':
        kernel_list = [gpflow.kernels.Exponential(lengthscales=l_scale - (i + 6) * 0.01)*gpflow.kernels.Periodic(base_kernel=gpflow.kernels.SquaredExponential(lengthscales=l_scale), period =3.) for i in range(latent_dim)]
        [set_untrainable_variance(k) for k in kernel_list]
        kernel = SeparateIndependent(kernel_list)
    elif kernel_type == 'partially_shared':
        #*gpflow.kernels.Periodic(base_kernel=gpflow.kernels.SquaredExponential(lengthscales=l_scale + i * 0.01), period =3.)
        kernel_list =[gpflow.kernels.Exponential(lengthscales=l_scale)* gpflow.kernels.Periodic(base_kernel=gpflow.kernels.SquaredExponential(lengthscales=l_scale), period =1.0) for i in range(D)]
        [set_untrainable_variance(k) for k in kernel_list]
        kernel = PartlySharedIndependentMultiOutput(kernel_list, nu =(nu +mnu_val))
    else:
        raise NotImplementedError
    return kernel

def set_untrainable_variance(kernel):
    if hasattr(kernel, 'kernels'):
        for k in kernel.kernels:
            if k.name =='product' or k.name== 'sum':
                set_untrainable_variance(k)
            elif k.name == 'periodic':
                gpflow.set_trainable(k.base_kernel.variance, False)
            else:
                gpflow.set_trainable(k.variance, False)
    else:
        if kernel.name == 'product' or kernel.name == 'sum':
            set_untrainable_variance(kernel)
        elif kernel.name == 'periodic':
            gpflow.set_trainable(kernel.base_kernel.variance, False)
        else:
            gpflow.set_trainable(kernel.variance, False)

def __MGARCH_data_preprocesssing(data):
    X,Y = data
    pd_rets = pd.DataFrame(Y)
    return pd_rets

def sample_y(mu, sigma):
    '''
    takes mu and sigma samples, samples y from multivariate gaussian and
    returns the mean and the variance of y
    :param mu samples of mu (SxNxD)
    :param sigma samples of sigma (SxNxDxD)
    :return mean (NxD) and variance (NxD) of y
    '''
    y_samples = np.zeros_like(mu)
    S,N,_ = y_samples.shape
    for s in range(S):
        for t in range(N):
            y_samples[s, t] = np.random.multivariate_normal(mu[s, t], sigma[s, t])
    mu_y = np.mean(y_samples, axis=0)
    var_y = np.var(y_samples, axis=0)
    return mu_y, var_y, y_samples

def cross_validate(modelname, data, n_splits = 4, test_s=None, num_iter = 500, num_samples = 200):
    '''
    :param modelname: model to be trained
    :param data: (X,Y) with X: N*P and Y: N*PxD
    :param n_splits: the number of test splits
    :param test_s: maximum test block size, meaning how many days ahead are predicted at max
    :return: list of predictions made by the model
    '''
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size= test_s)
    X,Y = data
    T = np.max(X)
    N,D = Y.shape
    i = 0

    test_all = []
    pred_all = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = np.take(X, train_index), np.take(X,test_index)
        y_train, y_test = np.take(Y, train_index, axis = 0), np.take(Y,test_index, axis = 0)
        print(f"round {i} train size {X_train.shape} test size {X_test.shape} ")
        i = i + 1
        test_all.append(y_test)
        #train model
        if modelname == "BANNER":
            #TODO num inducing 0.2 etc en batch size
            wishart_model = run_BANNER(data=(X, Y), mnu = "shared", T=T,iterations=num_iter,num_inducing=int(0.4*N),batch_size=100)
            posterior_wishart_process = wishart_model['wishart process']
            tiled_testpoints = np.tile(X_test, (D, 1)).T
            sigma_samples_gwp , mu_samples_gwp= posterior_wishart_process.predict_mc(tiled_testpoints, num_samples)
            y_mean_gwp, y_var_gwp, y_samples = sample_y(mu_samples_gwp, sigma_samples_gwp)
            pred_all.append(y_samples)
            #samples

        elif modelname == "MOGP":
            mogp_model = run_MOGP(data=(X,Y), iterations = 100)[0]
            aug_test_X, aug_test_y = format_data(X_test, y_test)
            mu_y,var_y = mogp_model.predict_f(np.array(aug_test_X))
            y_samples = mogp_model.predict_f_samples(np.array(aug_test_X),num_samples)
            pred_all.append(y_samples)
            #samples
        elif modelname == "MGARCH":
            ntrain = X_train.shape[0]
            ntest = X_test.shape[0]
            mgarch_mu, mgarch_sigma = run_MGARCH((X,Y),ntrain,ntest)
            pred_all.append(mgarch_mu)
        else:
            raise Exception("Sorry this model type is not recognised, try BANNER, MOGP or MGARCH")
    #for nr of test rounds
        #divide the data into training data and test data (training must have at least 100 data points)
        #train the model
        #predict
    #average mse/correlation?
    #if we take the average, we lose some information about where the model struggled the most.
    #perhaps return the entire list so at least we can plot the thing
    #also return the predicted points, so we can plot them on top of our data series.
    if modelname == "MGARCH":
        ys= np.concatenate((test_all), axis= 0)
        ps= np.concatenate((pred_all),axis = 0)
    else:
        ys = np.concatenate((test_all), axis=0)
        ps = np.concatenate((pred_all), axis=1)
    return ys,ps,