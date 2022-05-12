import sklearn as sk

from sklearn.model_selection import TimeSeriesSplit
from AnalysisUtils import run_inference_pcg as rn
import numpy as np
from AnalysisUtils import data_processing_pcg as datapcg

data, names = datapcg.import_ESMdata()
print(data.head(n=5))
print(data)
