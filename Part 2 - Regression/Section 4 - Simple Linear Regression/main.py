# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model,metrics

#load the boston dataset
boston = datasets.load_boston(return_X_y = False)

# defining feature matrix(X) and response vector(y) 
X = boston.data
y = boston.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4, random_state = 1)


reg = linear_model.LinearRegression()
reg.fit(X,y)

