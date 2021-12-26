import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold

df = pd.read_csv(r"Add your own file path/file.csv")
df.head()

print(df.Class.value_counts())     # No Fraud = 0 ; Fraud = 1

sns.countplot('Class', data=df)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)

rob_scaler = RobustScaler()
df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_Time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
df.drop(['Amount', 'Time'], axis = 1)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]
normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)
new_df.head()

print(new_df.Class.value_counts())

x = new_df.drop('Class', axis = 1)
y = new_df['Class']

model = ExtraTreesClassifier()
model.fit(x, y)
feat_importances = pd.Series(model.feature_importances_, index = x.columns)
feat_importances.nlargest(30).plot(kind = 'barh')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
print('xtrain is: ', x_train.shape)
print('xtest is: ', x_test.shape)
print('ytrain is: ', y_train.shape)
print('ytest is: ', y_test.shape)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

acc = accuracy_score(y_pred, y_test)
print(acc)
