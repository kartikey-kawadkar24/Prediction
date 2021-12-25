import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

df = pd.read_csv(r"Select your own path/file.csv")   #Please change the path while executing
df.head()

df.info()

df.isnull()

le_Gender = LabelEncoder()
le_CLASS = LabelEncoder()
df['CLASS'] = le_CLASS.fit_transform(df['CLASS'])
df['Gender'] = le_Gender.fit_transform(df['Gender'])

x = df.drop('CLASS', axis = 1)
y = df['CLASS']

model = ExtraTreesClassifier()
model.fit(x,y)
feat_importance = pd.Series(model.feature_importances_, index = x.columns)
feat_importance.nlargest(10).plot(kind = 'barh')
plt.show()

X = df[['HbA1c', 'BMI', 'AGE', 'Chol', 'TG', 'VLDL']]
y = df['CLASS']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print("x_train is :", x_train.shape)
print("x_test is :", x_test.shape)
print("y_train is :", y_train.shape)
print("y_test is :", y_test.shape)

model = XGBClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

acc = accuracy_score(y_test, y_pred)
print(acc * 100, '%')
