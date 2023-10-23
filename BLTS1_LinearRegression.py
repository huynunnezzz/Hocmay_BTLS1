import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

print("\nLinearRegression")
reg = LinearRegression()
reg.fit(X_train, Y_train)
y_preLR = reg.predict(X_test)
print("Chenh lech %.10f" % r2_score(Y_test,y_preLR))
print('MAE: ' , metrics.mean_absolute_error(Y_test, y_preLR))
print('NSE: ' , NSE(Y_test, y_preLR))
print('RMSE: ' , np.sqrt(metrics.mean_squared_error(Y_test, y_preLR)))

y = np.array(Y_test)
print("Thuc te - du doan = chech lech")
for i in range(0,len(y)):
    print(y[i],"-",y_preLR[i],"=",abs(y[i]-y_preLR[i]))









