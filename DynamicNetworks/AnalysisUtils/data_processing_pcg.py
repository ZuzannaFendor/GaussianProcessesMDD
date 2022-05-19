import numpy as np
import pandas as pd

def import_ESMdata():
    '''
    this function loads the data and performs the initial basic processing
    :return: scl dataset and daily dataset
    '''
    data = pd.read_csv(r"C:\Users\Zizi\Desktop\master\Thesis\GaussianProcessesMDD\ESMdata\ESMdata.csv")
    # day no is the day number of the year in which the experiment took place, so
    # it starts at 226 and ends at 98.
    # this is shifted to start at 1 and end at 239 such that the first day is day 1 and the last day is day 239
    data["dayno"] = data["dayno"].apply(lambda x: x - 225 if x in range(225, 367) else x + 141)
    scl_data = extract_scl(data)
    data_indeces = [i for i in range(2,6+1)]+[i for i in range(10,data.columns.get_loc("SCL.90.R.14"))]
    data = data.iloc[:,data_indeces]
    data['hour_no'] = data.apply(lambda row: dayno_to_hour(row), axis=1)
    return data, data.columns.values.tolist(), scl_data

def extract_scl(data):
    scl_indeces = [2, 3, 4] + [i for i in range(data.columns.get_loc("SCL.90.R.29"), (data.columns.get_loc("dep") + 1))]
    scl_data = data.iloc[:, scl_indeces]
    scl_data = scl_data.dropna()
    scl_data = scl_data.drop_duplicates("dayno")
    return scl_data, scl_data.columns.values.tolist()

def dayno_to_hour(row):
    beeptime = row["beeptime"]
    beephour = int(beeptime[:2])
    beepminute = int(beeptime[3:])
    beepday = int(row["dayno"])
    if beepminute <30:
        return beepday*24 + beephour
    else:
        return beepday*24 + beephour+1


def remove_incomplete_rows(data):

    pass

def intrapolate_nearest(X,Y):

    pass

def intrapolate_average(data):
    res = data.resample('s').interpolate().resample('15T').asfreq().dropna()
    pass

def interpolate_remove(data):
    pass

def stackify_data(X,Y):
    N,D = Y.shape
    X_aug = []
    Y_aug = []
    for n in range(N):
        for d in range(D):
            if not np.isnan(Y[n,d]) :
                Y_aug.append([Y[n,d], d])
                X_aug.append([X[n], d])
    return X_aug, Y_aug