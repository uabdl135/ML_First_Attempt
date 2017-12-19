    # -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 01:00:59 2017
    
@author: Abdul
    
Polynomial regression
    
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
    
data1=pd.read_csv('Position_Salaries.csv')
X=data1.iloc[:,1:2]#to give it a matrix appearance
Y=data1.iloc[:,2]
    
from sklearn.linear_model import LinearRegression
l_regressor = LinearRegression()
l_regressor.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
poly_X = poly.fit_transform(X)
poly.fit(poly_X,Y)
l1_regressor = LinearRegression()
l1_regressor.fit(poly_X,Y)
    
plt.scatter(X,Y,color='red',linewidths=20)
plt.plot(X,l_regressor.predict(X),color='blue')
plt.title('TRUTH OR BLUFF " LINEAR REGRESSION')
plt.xlabel('position')
plt.ylabel('Salary')
plt.show()
    
'''plt.scatter(X,Y,color='red')
plt.plot(X,l1_regressor.predict(poly.fit_transform(poly_X)),color='blue')
plt.show()'''
