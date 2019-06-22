# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

dataset = pd.read_csv('Advertising.csv')

dataset = dataset.drop(['Unnamed: 0'], axis = 1)


from sklearn.linear_model import LinearRegression

X = dataset.drop(['sales'], axis = 1)
y = dataset['sales'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(X,y)

print("The linear models is: Y = {:.5} + {:.5}*TV + {:.5}*radio + {:.5}*newspaper".format(reg.intercept_[0], reg.coef_[0][0],reg.coef_[0][1],reg.coef_[0][2]))

import statsmodels.api as sm

X_data = np.column_stack((dataset['TV'], dataset['radio'], dataset['newspaper']))
y = dataset['sales']

X2 = sm.add_constant(X_data)
est = sm.OLS(y,X2)
est2 = est.fit()
print(est2.summary)