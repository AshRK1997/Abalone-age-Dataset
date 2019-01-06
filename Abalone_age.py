#Solution to Prediction of Abalone age in HackerEarth
#url to dataset : https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data
#hackerearth link to question : https://www.hackerearth.com/problem/approximate/predict-abalone-age/

import pandas as pd
import numpy as np
from sklearn import linear_model
M = int(input())


features = ['sex', 'length', 'diameter', 'height', 'wheight', 'sweight', 'vweight', 'shweight', 'rings']

train = pd.DataFrame(columns=features, index=[])

temp = {}
for i in range(M):
    d = input()
    d = d.split()
    for j,c in zip(range(9),features):
        temp[c] = d[j]
    train.loc[str(i)] = pd.Series(temp)

b = train

c = b.filter(train.columns[[8]], axis=1)
train.drop(train.columns[[8]], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
le1.fit(train.sex)
train.sex = le1.transform(train.sex)

from sklearn.preprocessing import OneHotEncoder

x = np.array(train)
y = np.array(c)

ohe = OneHotEncoder(categorical_features=[0])

ohe.fit(x)

x = ohe.transform(x).toarray()



N = int(input())

test = pd.DataFrame(columns = features[:-1], index=[])
from sklearn import linear_model
temp = {}
for i in range(N):
    d = input()
    d = d.split()
    for j,c in zip(range(8),features[:-1]):
        temp[c] = d[j]
    test.loc[str(i)] = pd.Series(temp)
from sklearn.preprocessing import LabelEncoder

le2 = LabelEncoder()
le2.fit(test.sex)
test.sex = le1.transform(test.sex)

from sklearn.preprocessing import OneHotEncoder

xts = np.array(test)

ohe = OneHotEncoder(categorical_features=[0])

ohe.fit(xts)

xts = ohe.transform(xts).toarray()

reg = linear_model.LinearRegression()
reg.fit(x,y)

prediction = reg.predict(xts)

for i in range(N):
    print (int(prediction[i]+1.5))

