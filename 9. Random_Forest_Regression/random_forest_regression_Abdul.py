# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 21:57:53 2017

@author: Abdul
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataforest = pd.read_csv('Position_Salaries.csv')
X=dataforest.iloc[: ,1:2].values
Y=dataforest.iloc[: , 2].values

from sklearn.ensemble import RandomForestRegressor

