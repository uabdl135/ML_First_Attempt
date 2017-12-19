# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:20:28 2017

@author: Abdul
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: , 1:2].values
Y = dataset.iloc[: , 2].values

from sklearn.tree import DecisionTreeRegressor
dtree= DecisionTreeRegressor(random_state=0)
dtree.fit(X,Y)

y_pred = dtree.predict(6.5)

#smoother curve
X_grid=X
X_grid = np.arange( min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,dtree.predict(X_grid),color='blue')
plt.title('Regression using Decision tree')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

