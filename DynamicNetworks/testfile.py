import sklearn as sk

from sklearn.model_selection import TimeSeriesSplit
from AnalysisUtils import run_inference_pcg as rn
import numpy as np
import pandas as pd
from AnalysisUtils import data_processing_pcg as datapcg

data, names, scl = datapcg.import_ESMdata()

my_selection = data[['mood_relaxed',
       'mood_down', 'mood_irritat', 'mood_satisfi', 'mood_lonely',
       'mood_anxious', 'mood_enthus', 'mood_suspic', 'mood_cheerf',
       'mood_guilty', 'mood_doubt', 'mood_strong', 'pat_restl', 'pat_agitate',
       'pat_worry', 'pat_concent', 'se_selflike', 'se_ashamed', 'se_selfdoub',
       'se_handle','phy_hungry', 'phy_tired', 'phy_pain', 'phy_dizzy',
       'phy_drymouth', 'phy_nauseous', 'phy_headache', 'phy_sleepy']]

my_selection= my_selection.dropna()

data1 = pd.read_csv(r"C:\Users\Zizi\Desktop\master\Thesis\GaussianProcessesMDD\ESMdata\ESMdata.csv")
data1.loc[:,'date'] = pd.to_datetime(data1.date.astype(str)+' '+data1.beeptime.astype(str))
dat = data1.set_index(pd.DatetimeIndex(data1["date"])).drop("date", axis = 1)
print(dat.head(20))
dat = dat[['mood_relaxed',
       'mood_down', 'mood_irritat', 'mood_satisfi', 'mood_lonely',
       'mood_anxious', 'mood_enthus', 'mood_suspic', 'mood_cheerf',
       'mood_guilty', 'mood_doubt', 'mood_strong', 'pat_restl', 'pat_agitate',
       'pat_worry', 'pat_concent', 'se_selflike', 'se_ashamed', 'se_selfdoub',
       'se_handle','phy_hungry', 'phy_tired', 'phy_pain', 'phy_dizzy',
       'phy_drymouth', 'phy_nauseous', 'phy_headache', 'phy_sleepy']]
print("nan containing data",dat[dat.isna().any(axis=1)])
ind = dat.isna().any(axis=1)
dat = dat.interpolate(method="linear")
print("hopefully fixed", dat[ind])
print("any nana",dat[dat.isna().any(axis=1)])
data3 = dat.resample("H").ffill()#interpolate(method="linear")
# print("any nana",dat[dat.isna().any(axis=1)])
#
# print(data3.head(20))