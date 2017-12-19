# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:2].values
Y=dataset.iloc[:, 2:3].values

#feature scaling as svr doesn't have it by default
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_Y=StandardScaler()
X=sc_X.fit_transform(X)
Y=sc_Y.fit_transform(Y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf') #default is rbf too
regressor=regressor.fit(X,Y)
#no such paramter regressor.transform(X)

Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='green')
plt.title('Regression using SVR')
plt.xlabel('POSITION')
plt.ylabel('Salary')
plt.show()

