import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics


def NSE(y_test, y_pred):
    SSE = np.sum((y_test - y_pred) ** 2)
    SST = np.sum((y_test - np.mean(y_test)) ** 2)
    nse_value = 1 - (SSE / SST)
    return nse_value

data = pd.read_csv('traintaxippff.csv')
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False)

def error(y, y_pred):
    sum = 0
    for i in range(0, len(y)):
        sum = sum + abs(y[i] - y_pred[i])
    return sum / len(y)

min = 999999
kf = KFold(n_splits=5, random_state=None)

for train_index, validation_index in kf.split(dt_Train):
    X_train, X_validation = dt_Train.iloc[train_index, :7], dt_Train.iloc[validation_index, :7]
    y_train, y_validation = dt_Train.iloc[train_index, 7], dt_Train.iloc[validation_index, 7]

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    y_validation_pred = reg.predict(X_validation)
    y_train = np.array(y_train)
    y_validation = np.array(y_validation)

    sum_error = error(y_train, y_train_pred) + error(y_validation, y_validation_pred)
    if (sum_error < min):
        min = sum_error
        regr = reg

X_test = dt_Test.iloc[:, :7]
y_test = dt_Test.iloc[:, 7]
y_test_pred = regr.predict(X_test)
print("Chenh lech  %.10f" % r2_score(y_test, y_test_pred))
print('MAE: ' ,metrics.mean_absolute_error(y_test, y_test_pred))
print('NSE: ', NSE(y_test, y_test_pred))
print('RMSE: ' , np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

y = np.array(y_test)
print("Thuc te - Du doan = Chenh lech")
for i in range(0, len(y)):
    print(y[i], " - ", y_test_pred[i], " = ", abs(y[i] - y_test_pred[i]))