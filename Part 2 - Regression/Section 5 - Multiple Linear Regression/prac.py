# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Advertising.csv')

dataset = dataset.drop(['Unnamed: 0'], axis = 1)

#plt.figure(figsize = (16,8))
plt.scatter(dataset['TV'],dataset['sales'], color = 'black')
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()


X = dataset['TV'].values.reshape(-1,1)
y = dataset['sales'].values.reshape(-1,1)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X,y)

print("Linear Model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

predictions = reg.predict(X)

plt.scatter(dataset['TV'], dataset['sales'], color = 'blue')
plt.plot(dataset['TV'], predictions, color = 'red', linewidth = 2)
plt.xlabel('Money spent on TV ads ($)')
plt.ylabel('Sales ($)')
plt.show()

import statsmodels.api as sm

X2 = sm.add_constant(X)
est = sm.OLS(y,X2)
est2 = est.fit()






