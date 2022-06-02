import sklearn as sk

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # for PCA calculation
from AnalysisUtils import run_inference_pcg as rn
import numpy as np
import pandas as pd
from AnalysisUtils import data_processing_pcg as datapcg
from AnalysisUtils import run_inference_pcg as runpcg
import tensorflow as tf
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


data, cols, scl = datapcg.import_ESMdata()
my_selection = data[['mood_relaxed','mood_down','mood_irritat', 'mood_satisfi', 'mood_lonely',
       'mood_anxious', 'mood_enthus', 'mood_suspic', 'mood_cheerf',
       'mood_guilty', 'mood_doubt', 'mood_strong','pat_agitate']]
# my_selection = data[['mood_relaxed',
#        'mood_down', 'mood_irritat', 'mood_satisfi', 'mood_lonely',
#        'mood_anxious', 'mood_enthus', 'mood_suspic', 'mood_cheerf',
#        'mood_guilty', 'mood_doubt', 'mood_strong', 'pat_restl', 'pat_agitate',
#        'pat_worry', 'pat_concent', 'se_selflike', 'se_ashamed', 'se_selfdoub',
#        'se_handle','phy_hungry', 'phy_tired', 'phy_pain', 'phy_dizzy',
#        'phy_drymouth', 'phy_nauseous', 'phy_headache', 'phy_sleepy']]

my_selection= my_selection.dropna()
X = my_selection.values
sc = StandardScaler()
X_std = sc.fit_transform(X)
cor = pd.DataFrame(np.corrcoef(X_std, rowvar=False), columns = my_selection.columns.values)
cor.index =my_selection.columns.values


pca = PCA()
X_pca = pca.fit(X_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

print(f"explained variance ratio {pca.explained_variance_ratio_}")
n_pcs = pca.components_.shape[0]
pc_comp = (pd.DataFrame(pca.components_, columns = my_selection.columns))
print(pc_comp)
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names = my_selection.columns.values
# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

# LIST COMPREHENSION HERE AGAIN
dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
print(dic)
print("finished")