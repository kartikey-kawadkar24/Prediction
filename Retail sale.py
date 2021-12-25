import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"Add your own path/file.csv")
df.head()

df.info()

le_Area = LabelEncoder()

le_Purchase_Channel = LabelEncoder()
df['Area'] = le_Area.fit_transform(df['Area'])
df['Purchase Channel'] = le_Purchase_Channel.fit_transform(df['Purchase Channel'])

le_Spend_Category = LabelEncoder()
df['Spend Category'] = le_Spend_Category.fit_transform(df['Spend Category'])

x = df.drop('Sale Made', axis = 1)
y = df['Sale Made']

model = ExtraTreesClassifier()
model.fit(x,y)
feat_importance = pd.Series(model.feature_importances_, index = x.columns)
feat_importance.nlargest(10).plot(kind = 'barh')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
print("shape of xtrain is :", x_train.shape)
print("shape of ytrain is :", y_train.shape)
print("shape of xtest is : ", x_test.shape)
print("shape of ytest is :", y_test.shape)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

acc = accuracy_score(y_test, y_pred)
print(acc)
