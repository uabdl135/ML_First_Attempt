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
    X=data1.iloc[:,1:2].values#to give it a matrix appearance
    Y=data1.iloc[:,2].values
    
    from sklearn.linear_model import LinearRegression
    l_regressor = LinearRegression()
    l_regressor.fit(X,Y)
    
    from sklearn.preprocessing import PolynomialFeatures
    poly=PolynomialFeatures(degree=4)
    poly_X = poly.fit_transform(X)
    poly.fit(poly_X,Y)
    l1_regressor = LinearRegression()
    l1_regressor.fit(poly_X,Y)
    
    plt.scatter(X,Y,color='red')
    plt.plot(X,l_regressor.predict(X),color='blue')
    plt.title('TRUTH OR BLUFF  LINEAR REGRESSION')
    plt.xlabel('position')
    plt.ylabel('Salary')
    plt.show()
    
    plt.scatter(X,Y,color='red')
    plt.plot(X,l1_regressor.predict(poly.fit_transform(X)),color='blue')
    plt.title('TRUTH OR BLUFF  POLYNOMIAL REGRESSION')
    plt.xlabel('position')
    plt.ylabel('Salary')
    plt.show()
    
    
    Xgrid=np.arange(min(X),max(X),0.1)
    Xgrid.reshape(len(Xgrid),1)
    plt.scatter(X,Y,color='red')
    plt.plot(X_grid,l1_regressor.predict(poly.fit_transform(X_grid)),color='green')
    plt.title("IN range 0.1")
    plt.xlabel("Position")
    plt.ylabel("Salary")
    plt.show()
    
    l1_regressor.predict(poly.fit_transform(6.5))