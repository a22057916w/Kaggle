import pandas as pd
import numpy as np
import os

absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

print(absolutepath)
print(fileDirectory)

train = pd.read_csv(os.path.join(fileDirectory, "train.csv"))

#print(type(train.shape))
#print(train.shape)

df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4], 'col3': [5, np.NaN], 'col4': [np.NaN, 8]})
print(df)
print(df.shape)
#print(type(df.iloc[0].values))
#print(df.iloc[:, 2])

#print(type(df.iloc[0]))
for i in range(df.shape[1]):
    if(df.iloc[:,i].isnull().values.any()):
        print("{0} idx is {1} that has {2} null values".format(df.columns[i], i, df.iloc[:,i].isnull().sum()))
