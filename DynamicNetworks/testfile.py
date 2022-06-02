import sklearn as sk

from sklearn.model_selection import TimeSeriesSplit
from AnalysisUtils import run_inference_pcg as rn
import numpy as np
import pandas as pd
from AnalysisUtils import data_processing_pcg as datapcg
from AnalysisUtils import run_inference_pcg as runpcg
import tensorflow as tf
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data, names, scl = datapcg.import_ESMdata()

my_selection = data[['mood_relaxed',
       'mood_down', 'mood_irritat', 'mood_satisfi', 'mood_lonely',
       'mood_anxious', 'mood_enthus', 'mood_suspic', 'mood_cheerf',
       'mood_guilty', 'mood_doubt', 'mood_strong', 'pat_restl', 'pat_agitate',
       'pat_worry', 'pat_concent', 'se_selflike', 'se_ashamed', 'se_selfdoub',
       'se_handle','phy_hungry', 'phy_tired', 'phy_pain', 'phy_dizzy',
       'phy_drymouth', 'phy_nauseous', 'phy_headache', 'phy_sleepy']]

my_selection= my_selection.dropna()

datainit = pd.read_csv(r"C:\Users\Zizi\Desktop\master\Thesis\GaussianProcessesMDD\ESMdata\ESMdata.csv")

datainit['date'] = pd.to_datetime(datainit.date.astype(str) + ' ' + datainit.beeptime.astype(str) ,  format = "%d/%m/%y %H:%M")
data = datainit[['mood_relaxed','mood_down', 'mood_irritat', 'mood_satisfi', 'date']]

data = data.set_index(pd.DatetimeIndex(data["date"])).drop("date", axis = 1)
print(data.head(20))
# data = dat[['mood_relaxed',
#        'mood_down', 'mood_irritat', 'mood_satisfi']]
       # , 'mood_lonely',
       # 'mood_anxious', 'mood_enthus', 'mood_suspic', 'mood_cheerf',
       # 'mood_guilty', 'mood_doubt', 'mood_strong', 'pat_restl', 'pat_agitate',
       # 'pat_worry', 'pat_concent', 'se_selflike', 'se_ashamed', 'se_selfdoub',
       # 'se_handle','phy_hungry', 'phy_tired', 'phy_pain', 'phy_dizzy',
       # 'phy_drymouth', 'phy_nauseous', 'phy_headache', 'phy_sleepy']]

ind = data.isna().any(axis=1)
nrinf = ind.sum()
bb = data[ind]
dat = data.interpolate(method="linear")
aa = dat[ind]

data4 = dat.resample("3H").pad() #asfreq()
data5 = dat.resample("3H").nearest()
data6 = dat.resample("3H").first(min_count =1 ).interpolate(method="time")
data7 = dat.resample("3H").first(min_count =1 )
data3 = dat.resample("H").ffill()#interpolate(method = "slinear", order  =5)#interpolate(method="linear")#ffill()#
#
# print("any nana",data3[data3.isna().any(axis=1)])
# # data3 = data3.dropna()
# print("and now?",data3[data3.isna().any(axis=1)])
# #print()
data6 = dat.resample("H").sum()
print(data5.head(40))

# X = data['hour_no'].to_numpy(dtype="float64")[:200]
# Y = data.loc[:,['mood_down','pat_worry','phy_tired']].to_numpy(dtype="float64")[:200,:]
# d = (X,Y)
# model = runpcg.run_MGARCH(d)
# print(model["forecast"])
# print(model['covariance_matrix'])
