import numpy as np
import scipy

def MSE(Ytest, predictions ):
    '''

    :param Ytest: (N x D) the true data values
    :param predictions: (R x N x D) the predictions provided by the model
    :return: (R x N x D) R samples N timepoints, D dimensions the mean squared errors
    '''
    mse = np.zeros_like(predictions)
    for sample in range(predictions.shape[0]):
        for iy,y in enumerate(Ytest):
            mse[sample,iy] = (y - predictions[sample,iy])**2
    return mse

def correlation(Ytest, predictions):
    '''

    :param Ytest: (Ntest x D)
    :param predictions: (R x Ntest x D)
    :return: average correlation between the prediction and the test
    '''
    D = Ytest.shape[-1]
    avg_pred = np.mean(predictions, axis  = 0) #averaged over samples
    corr = np.corrcoef(Ytest,avg_pred, rowvar=False)
    corr = np.diag(corr[:D,D:])

    return corr

def corr_timeseries(a,b):
    q, p = scipy.stats.pearsonr(a, b)
    return q,p

def correlation_sigma(Ytest, predictions):
    '''
    Flattens the pridctions and labels such that each covariance of i j,
    is flattened into N x DD
    :param Ytest: (Ntest x D x D)
    :param predictions: (R x Ntest x D x D)
    :return: correlation averaged over R, for each n in N
    '''
    D = Ytest.shape[-1]
    N = Ytest.shape[0]

    avg_pred = np.reshape(np.mean(predictions, axis  = 0), (N,D*D) )#averaged over samples
    Ytest = np.reshape(Ytest,(N,D*D))
    corr = np.corrcoef(Ytest[0], avg_pred[0], rowvar = False)
    corr = np.diag(corr[:,:D*D,D*D:], axis = 1)

    return corr