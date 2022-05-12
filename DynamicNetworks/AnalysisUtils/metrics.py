import numpy as np

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