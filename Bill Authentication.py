import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r"Add your own file path/file.csv")
df.head()

features = df.loc[:, df.columns != 'Class'].values[:, 1:]
labels = df.loc[:, 'Class'].values
print(labels[labels == 1].shape[0], labels[labels ==0].shape[0])

scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(features)
y = labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
print("shape of xtrain is :", x_train.shape)
print("shape of ytrain is :", y_train.shape)
print("shape of xtest is : ", x_test.shape)
print("shape of ytest is :", y_test.shape)

model = SVC(kernel = 'linear')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

acc = accuracy_score(y_test, y_pred)
print(acc)
