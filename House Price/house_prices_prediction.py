import os
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
def fillna_nonnumerical(series):
    series = series.fillna(series.value_counts().idxmax())
    return series

def mapping(series):
    unique=series.unique()
    map = { unique[i] : i for i in range(len(unique)) }
    return series.map(map)

def plt_data_bar(series):
    X = series.value_counts().index
    Y = series.value_counts().values
    print(X,Y)
    fig, ax = plt.subplots(figsize=(10, 4))
    idx = np.asarray([i for i in range(len(X))])
    data = {key: val for key, val in zip(X, Y)}
    ax.bar(idx, [val for key,val in sorted(data.items())])
    ax.set_xticks(idx)
    ax.set_xticklabels(X)
    ax.set_xlabel(series.name)
    ax.set_ylabel('count')
    plt.show()

#get the relative path of dataset
absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)
#load data
train = pd.read_csv(fileDirectory + r'\train.csv')
test = pd.read_csv(fileDirectory + r'\test.csv')



#visualization
# plt_data_bar(train['MSSubClass'])
# plt_data_bar(train['MSZoning'])



#data preproccessing
print(train.shape)
#check null_value in series
for i in range(train.shape[1]):
    if(train.iloc[:,i].isnull().values.any()):
        print("{0} idx is {1} that has {2} null values".format(train.columns[i],i,train.iloc[:,i].isnull().sum()))

scaler = StandardScaler()
train['MSZoning'] = fillna_nonnumerical(train['MSZoning'])
train['MSZoning'] = mapping(train['MSZoning'])


