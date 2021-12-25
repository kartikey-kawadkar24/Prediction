import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"add your own file path")
df.head()

features = df.loc[:, df.columns != 'status'].values[:,1:]
labels = df.loc[:, 'status'].values
labels = df.status
# print(labels)
labels.head()

print(labels[labels == 1].shape[0], labels[labels == 0].shape[0])

scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(features)
y = labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
print("shape of xtrain is :", x_train.shape)
print("shape of ytrain is :", y_train.shape)
print("shape of xtest is : ", x_test.shape)
print("shape of ytest is :", y_test.shape)

model = XGBClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
