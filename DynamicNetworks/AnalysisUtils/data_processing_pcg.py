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
    data['date'] = pd.to_datetime(data.date.astype(str) + ' ' + data.beeptime.astype(str),
                                      format="%d/%m/%y %H:%M")
    data = data.set_index(pd.DatetimeIndex(data["date"])).drop("date", axis=1)
    data["dayno"] = data["dayno"].apply(lambda x: x - 225 if x in range(225, 367) else x + 141)
    scl_data = extract_scl(data)
    data_indeces = [i for i in range(2,6+1)]+[i for i in range(9,data.columns.get_loc("SCL.90.R.14"))]
    data = data.iloc[:,data_indeces]
    data['hour_no'] = data.apply(lambda row: dayno_to_hour(row), axis=1)

    #TODO all likert scales to 0-7 scale instead of -3 to 3 in some cases.
    #down, lonely, anxious, guilty
    print(data.columns.values)
    print(min(data["mood_down"]))
    data["mood_down"] = data["mood_down"] + 4
    data["mood_lonely"] = data["mood_lonely"] + 4
    data["mood_guilty"] = data["mood_guilty"] + 4
    data["mood_anxious"] = data["mood_anxious"] + 4
    print(min(data["mood_down"]))
    data = categorize_vars(data)
    data.dropna()
    return data, data.columns.values.tolist(), scl_data

def limit_vars(data):
    return data[['mood_relaxed', 'mood_down', 'mood_irritat', 'mood_satisfi', "dayno"]]

def categorize_vars(data):
    new_data = pd.DataFrame()
    new_data["neg_affect"] = data[["mood_down","mood_lonely","mood_guilty","mood_anxious","mood_doubt"]].sum(axis=1)
    new_data["pos_affect"] = data[["mood_satisfi", "mood_cheerf", "mood_relaxed","mood_enthus", "mood_strong" ]].sum(axis=1)
    new_data["mental_unrest"] = data[["mood_irritat", "pat_agitate"]].sum(axis=1)
    new_data["worry"] = data["pat_worry"]
    new_data["sus"] = data["mood_suspic"]
    return new_data


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



def intrapolate(data, method, samplesize = "3H"):
    '''
    resamples the data into equal intervals and fills in the gaps
    through choosing the nearest value or through linear interpolation
    :param data: dataset
    :param method: ["nearest"|"linear"] method of replacing the NaN values.
    :param samplesize:
    :return:
    '''
    if method=="nearest":
        data_res = data.resample(samplesize).pad()
    elif method == "linear":
        data_res = __intrapolate_linear(data, samplesize=samplesize)
    else:
        print("UNKNOWN INTERPOLATION")
    return data_res

def __intrapolate_linear(data, samplesize = "3H"):
    '''
    Resamples the data into new groups, takes the first of existing values within a group and fills in the
    blanks using linear interpolation
    :param data:
    :param samplesize: str, the sample period
    :return:
    '''

    return data.resample(samplesize ).first(min_count =1 ).interpolate(method="time")

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