import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv('C:/Users/USER/Desktop/Project/Datasets/banana_dataset.csv')
data

data["Gas"].fillna(data['Gas'].mean(), inplace=True) 

data.rename(columns= {'created_at': 'date'}, inplace=True)
data['date'] = data['date'].apply(lambda x: str(x)[:19])
data['date'] = pd.to_datetime(data['date'])

data["Grade"] = [0 for i in range (2245)]

data.loc[data['Gas'] < 130, 'Grade'] = 1
data.loc[(data['Gas'] >= 130) & (data['Gas'] < 210), "Grade"] = 2
data.loc[(data['Gas'] >= 210) & (data['Gas'] < 360), 'Grade'] = 3
data.loc[data['Gas'] >= 360, 'Grade'] = 4

import datetime
today = datetime.datetime.today()
today

data['date'] = pd.to_datetime(data['date'])
data['Hours'] = (today - data['date'])/np.timedelta64(1,'h')

lst = list(data['Hours'])
lst.sort()
data.drop('Hours', axis=1, inplace=True)
ser = pd.Series(data = lst, index = data.index)
data['Hours'] = ser
data


X = data[['Temperature','Gas']]
y = data['Hours']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


svr = SVR()
svr.fit(X_train, y_train)
pred_svr = svr.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_svr)))
r2_score(y_test, pred_svr)

plt.figure(figsize=(11,7))
plt.title('SVR')
sns.set_style('whitegrid')
plt.xlabel('Y Predictions')
plt.ylabel('Y True')
plt.scatter(pred_svr, y_test, color='r')

dtree = DecisionTreeRegressor()
dtree.fit(X_train, y_train)
pred_dtree = dtree.predict(X_test)

score = dtree.score(X_test, y_test)
print("R-squared: ", score)
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_dtree)))
print("R2: ", r2_score(y_test,pred_dtree))

plt.figure(figsize=(11,7))
plt.title('Decision Tree')
plt.xlabel('Y Predictions')
plt.ylabel('Y True')
sns.set_style('whitegrid')
plt.scatter(pred_dtree, y_test, color='g')

rfr = RandomForestRegressor(n_estimators = 1)
rfr.fit(X_train, y_train)
pred_rfr = rfr.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_rfr)))
r2_score(y_test, pred_rfr)

plt.figure(figsize=(11,7))
plt.title('RFR')
plt.xlabel('Y Predictions')
plt.ylabel('Y True')
sns.set_style('whitegrid')
plt.scatter(pred_rfr, y_test, color='brown')


