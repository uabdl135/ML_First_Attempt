# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:01:18 2017

@author: Abdul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading data
dataset1 = pd.read_csv('Salary_Data.csv')
X=dataset1.iloc[:,:-1].values
Y=dataset1.iloc[:,1].values

from sklearn.cross_validation import train_test_split 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33 , random_state=0)

#Fitting Simple Linear Regression Model to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)

# plotting for train set 
#plt.scatter(X,Y,color='green')
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs. Years_of_Experience(Training_Set)')
plt.xlabel('Years of Expereince')
plt.ylabel('Salary(In $)')
plt.show()

#testing for new set of data for test set

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs. Years_of_Experience(Test_Set) ')
plt.xlabel('Years of Expereince')
plt.ylabel('Salary(In $)')
plt.show()