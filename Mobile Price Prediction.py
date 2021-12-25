import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier

df = pd.read_csv(r"Add your own file path.file.csv")
df.head()

features = df.loc[:, df.columns != 'price_range'].values[:,1:]
labels = df.loc[:, 'price_range'].values
labels = df.price_range
labels.head()

print(labels[labels == 1].shape[0], labels[labels == 2].shape[0])

x = df.iloc[:, 0:20]
y = df.iloc[:, -1]

model = ExtraTreesClassifier()
model.fit(x, y)
feat_importances = pd.Series(model.feature_importances_, index = x.columns)
feat_importances.nlargest(10).plot(kind = 'barh')
plt.show()

X = df[['ram', 'battery_power', 'px_width', 'px_height', 'int_memory', 'touch_screen', 'wifi']]
y = df['price_range']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print("shape of xtrain is :", x_train.shape)
print("shape of ytrain is :", y_train.shape)
print("shape of xtest is : ", x_test.shape)
print("shape of ytest is :", y_test.shape)

model = linear_model.LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
