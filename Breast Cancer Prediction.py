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

df = pd.read_csv(r"add your own path/file.csv")
df.head()

df.info()

x = df.drop('Classification', axis = 1)
y = df['Classification']

model = ExtraTreesClassifier()
model.fit(x,y)
feat_importance = pd.Series(model.feature_importances_, index = x.columns)
feat_importance.nlargest(10).plot(kind = 'barh')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.2)
print("x_train is :", x_train.shape)
print("x_test is :", x_test.shape)
print("y_train is :", y_train.shape)
print("y_test is :", y_test.shape)

model = GradientBoostingClassifier() 
model.fit(x_train , y_train)

y_pred = model.predict(x_test)
print(y_pred)

acc = accuracy_score(y_pred, y_test)
print(acc * 100, '%')
