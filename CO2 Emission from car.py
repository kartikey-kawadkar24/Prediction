import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model

df = pd.read_csv(r"Add your own file path/file.csv")
df.head()

le_Car = LabelEncoder()
le_Model = LabelEncoder()
df['Car'] = le_Car.fit_transform(df['Car'])
df['Model'] = le_Model.fit_transform(df['Model'])

x = df.iloc[:, 0:4]
y = df.iloc[:, -1]

model = ExtraTreesClassifier()
model.fit(x,y)
feat_importance = pd.Series(model.feature_importances_, index = x.columns)
feat_importance.nlargest(5).plot(kind = 'barh')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
print("shape of xtrain is :", x_train.shape)
print("shape of ytrain is :", y_train.shape)
print("shape of xtest is : ", x_test.shape)
print("shape of ytest is :", y_test.shape)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)
