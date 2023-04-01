# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""rain_fall_prediction.py: Based of different given weather parameters, we can predict if it is going to rain or not."""

__author__      = 'Kartikey Kawadkar'

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score, roc_curve
from xgboost import XGBRegressor
from xgboost import xgboostClassifier


def rainfall_prediction_problem():
	trainDf = pd.read_csv("/home/kartik/kartik/rain_fall_prediction/dataset/train.csv")
	trainDf.fillna(trainDf.mean(), inplace = True)
	trainDf[['weather_char', 'weather_id']] = trainDf['weather_moment_id'].str.split("_", expand=True)


	trainDf = trainDf[['weather_char', 'weather_moment_id', 'weather_id', 'temperature(C)', 'wind_strength(kph)',
	'wind_angle(degree)', 'wind_direction', 'pressure(millibar)',
	'precipitation(mm)', 'relative_humidity(%)', 'feelslike_temp(C)',
	'wind_temp(C)', 'heatflow(C)', 'dewpoint(C)', 'vision(km)',
	'gust_strength(mph)', 'will_rain']]

	trainDf['weather_id'] = trainDf['weather_id'].astype(int)
	le_wind_direction = LabelEncoder()
	trainDf['wind_direction'] = le_wind_direction.fit_transform(trainDf['wind_direction'])

	x = trainDf.iloc[:, 2:15]
	y = trainDf.iloc[:, -1]

	model = ExtraTreesClassifier()
	model.fit(x,y)
	feat_importance = pd.Series(model.feature_importances_, index = x.columns)
	feat_importance.nlargest(10).plot(kind = 'barh')
	print(feat_importance)

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

	# model = linear_model.LinearRegression()
	# model = XGBRegressor()
	# model = LogisticRegression(max_iter = 5500)

	xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
	xgb_model.fit(x, y)

	y_pred = xgb_model.predict(x_test)
	print(y_pred)

	# print(confusion_matrix(y, y_pred))

	# model.fit(x_train, y_train)

	# y_pred = model.predict(x_test)
	# print(y_pred)

	# pred = model.predict([[1860, 25.2, 2.5, 1, 1009, 0.27, 85, 27.7, 25.2, 27.7, 22.6, 9, 2.5]])
	# pred = xgb_model.predict([['1860', '25.2', '2.5', '1', '1009', '0.27', '85', '27.7', '25.2', '27.7', '22.6', '9', '2.5']])
	pred = xgb_model.predict([[1860, 25.2, 2.5, 1, 1009, 0.27, 85, 27.7, 25.2, 27.7, 22.6, 9, 2.5]])
	print(pred)



rainfall_prediction_problem()