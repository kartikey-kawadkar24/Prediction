import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r"Add your own file path/file.csv")
df.head()

df.type = [1 if each == 'white' else 0 for each in df.type]

df.info()

df.isnull().sum()

df = df.fillna(df.mean())

df['best_quality'] = [1 if x >= 7 else 0 for x in df['quality']]

x = df.drop(['quality','best_quality'], axis = 1)
y = df['best_quality']

model = ExtraTreesClassifier()
model.fit(x, y)
feat_importances = pd.Series(model.feature_importances_, index = x.columns)
feat_importances.nlargest(22).plot(kind = 'barh')
plt.show()

df = df.drop('total sulfur dioxide', axis = 1)
df.head()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
print("shape of xtrain is :", x_train.shape)
print("shape of ytrain is :", y_train.shape)
print("shape of xtest is : ", x_test.shape)
print("shape of ytest is :", y_test.shape)

scal = MinMaxScaler()
scal_fit = scal.fit(x_train)
scal_xtrain = scal_fit.transform(x_train)
scal_xtest = scal_fit.transform(x_test)

model = RandomForestClassifier()
model.fit(scal_xtrain, y_train)

y_pred = model.predict(scal_xtest)

acc = accuracy_score(y_pred, y_test)
print(acc)
