import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn import metrics


data = pd.read_csv("traintaxippff.csv")
dttrain, dttest = train_test_split(data, test_size = 0.3, shuffle = False)
X_train = dttrain.iloc[:,:-1]
Y_train = dttrain.iloc[:, 7]
X_test = dttest.iloc[:, :-1]
Y_test = dttest.iloc[:, 7]

def NSE(y_test, y_pred):
    SSE = np.sum((y_test - y_pred) ** 2)
    SST = np.sum((y_test - np.mean(y_test)) ** 2)
    nse_value = 1 - (SSE / SST)
    return nse_value

# Lasso
print("\nLasso")
las = Lasso()
las.fit(X_train,Y_train)
y_preLs = las.predict(X_test)
print("Chenh lech %.10f" % r2_score(Y_test,y_preLs))
print('MAE: ' , metrics.mean_absolute_error(Y_test, y_preLs))
print('NSE: ' , NSE(Y_test, y_preLs))
print('RMSE: ' , np.sqrt(metrics.mean_squared_error(Y_test, y_preLs)))

#Ridge
print("\nRidge")
rid = Ridge()
rid.fit(X_train,Y_train)
y_preRd = rid.predict(X_test)
print("Chenh lech " , r2_score(Y_test,y_preRd))
print('MAE: ' , metrics.mean_absolute_error(Y_test, y_preRd))
print('NSE: ' , NSE(Y_test, y_preRd))
print('RMSE: ' , np.sqrt(metrics.mean_squared_error(Y_test, y_preRd)))


