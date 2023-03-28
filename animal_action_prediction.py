# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""animal_action_prediction.py: Given the action co-ordinates of body movements, we can predict the action."""

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
# from xgboost import XGBRegressor
# from xgboost import xgboostClassifier



def animal_action_prediction():
	trainDf = pd.read_csv("path_to_csv")
	print(trainDf.head())

	trainDf['instance_id_num'] = trainDf['Instance_ID'].str.replace('([A-Za-z]+)', '')
	# print(trainDf.columns)

	trainDf = trainDf[['Instance_ID', 'instance_id_num', 'Position_Part_1_x', 'Position_Part_1_y',
       'Position_Part_1_z', 'Anonymous_F1_Part_1', 'Anonymous_F2_Part_1',
       'Anonymous_F3_Part_1', 'Anonymous_F4_Part_1', 'Anonymous_F5_Part_1',
       'Anonymous_F6_Part_1', 'Position_Part_2_x', 'Position_Part_2_y',
       'Position_Part_2_z', 'Anonymous_F1_Part_2', 'Anonymous_F2_Part_2',
       'Anonymous_F3_Part_2', 'Anonymous_F4_Part_2', 'Anonymous_F5_Part_2',
       'Anonymous_F6_Part_2', 'Position_Part_3_x', 'Position_Part_3_y',
       'Position_Part_3_z', 'Anonymous_F1_Part_3', 'Anonymous_F2_Part_3',
       'Anonymous_F3_Part_3', 'Anonymous_F4_Part_3', 'Anonymous_F5_Part_3',
       'Anonymous_F6_Part_3', 'Position_Part_4_x', 'Position_Part_4_y',
       'Position_Part_4_z', 'Anonymous_F1_Part_4', 'Anonymous_F2_Part_4',
       'Anonymous_F3_Part_4', 'Anonymous_F4_Part_4', 'Anonymous_F5_Part_4',
       'Anonymous_F6_Part_4', 'Anonymous_W1', 'Anonymous_W2', 'Anonymous_W3',
       'Anonymous_W4', 'Anonymous_W5', 'Anonymous_W6', 'Anonymous_W7',
       'Anonymous_W8', 'Anonymous_W9', 'Anonymous_W10', 'Action',
       ]]
	print(trainDf.head())

	x = trainDf.iloc[:, 2:15]
	y = trainDf.iloc[:, -1]

	model = ExtraTreesClassifier()
	model.fit(x,y)
	feat_importance = pd.Series(model.feature_importances_, index = x.columns)
	feat_importance.nlargest(20).plot(kind = 'barh')
	print(feat_importance)

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

	print(x_train.shape)
	print(x_test.shape)
	print(y_train.shape)
	print(y_test.shape)

	model = linear_model.LinearRegression()
	model = XGBRegressor()
	model = LogisticRegression(max_iter = 5500)

	model.fit(x_train, y_train)

	y_pred = model.predict(x_test)
	print(y_pred)

	accuracy = accuracy_score(y_test, y_pred)
	print(accuracy)

	# print(x_test[0], y_pred)
	# xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
	# xgb_model.fit(x, y)

	# y_pred = xgb_model.predict(x_test)
	# print(y_pred)

	# print(confusion_matrix(y, y_pred))


animal_action_prediction()