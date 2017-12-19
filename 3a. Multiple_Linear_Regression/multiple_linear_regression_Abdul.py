# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:15:40 2017

@author: Abdul

Multiple linear regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('50_Startups.csv')
X=dataset.iloc[:, : -1].values
Y=dataset.iloc[:,4].values

#creating dummy variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
ohe = OneHotEncoder(categorical_features=[3])
X=ohe.fit_transform(X).toarray()

#avoiding dummy variable trap(although the libraries would have handled it)
X=X[:,1:]

from sklearn.cross_validation import train_test_split
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=0.2 , random_state =0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_X,train_Y)
Y_pred = regressor.predict(test_X)


'''BUILDING THE OPTIMAL MODEL USING BACKWARD ELIMINATION
    SL =0.05'''
import statsmodels.formula.api as sm
#addding A(0) in X or intercept 
X = np.append(arr=np.ones((50,1)).astype(int) , values =X , axis = 1)
#DO not know how to get the return p value , otherwise I would have used a loop obviously
X_opt= X[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog =Y,exog = X_opt).fit()
    #to get table containing data , especially the P-Value
regressor_ols.summary()

X_opt=X[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=X[:,[0,3,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=X[:,[0,3]]
regressor_ols=sm.OLS( endog = Y ,exog =X_opt).fit()
regressor_ols.summary()

from sklearn.cross_validation import train_test_split
train_X,test_X,train_Y,test_Y = train_test_split(X_opt,Y,test_size=0.2 , random_state =0)

'''from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_X,train_Y)
Y_pred_backward = regressor.predict(test_X)'''

