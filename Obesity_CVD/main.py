import sys
sys.path.append('D:\My Drive\ML2023\data-analysis')

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import math
from plotly.subplots import make_subplots


import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import pandas as pd 
from my_dash_class.my_plot import plot_, Correlation_Analisys
from my_dash_class.ak_learning import ak_learn,ak_clean,ak_filter

plt_ = plot_()
corr_an = Correlation_Analisys()
ak_clean = ak_clean()
ak_learn = ak_learn()
ak_filter = ak_filter()


df = ak_clean.df_get('ObesityDataSet.csv')

print(df.head())


# df_t= df[df.columns[[-1,9]]]
# df_c = df.drop(df.columns[[-1,9]], axis=1)
# df = pd.concat([df_c, df_t], axis=1)

# df = df.drop(df.columns[[0,4]], axis =1)
# print(df.head())




cmLabel,typOfVar,mapping,swapMapping = ak_clean.CleaningVar(df)
df = ak_clean.CleaningDF(df,typOfVar,mapping)
print(df.head())


cols = range(df.shape[1])
# mls = ['DT', 'KNN', 'SVC', 'NB', 'LR']

mls = ['knn','DT','lg','NB', 'ABC','RFC']
confidence_interval_limit = np.linspace(2,5,5)
correlation_percentage_threshold = np.linspace(.4,1,7)
pre_proc = ''
disp_dash = 'all'
mach = 'adv'
clf, scre_max, corr_per_max,corr_tmp_max,std_number_max, ml_max = ak_learn.Search_ML(df,mls,mach,pre_proc,confidence_interval_limit,correlation_percentage_threshold,disp_dash)

