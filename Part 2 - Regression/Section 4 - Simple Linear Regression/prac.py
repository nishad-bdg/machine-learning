# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(X_train,y_train)

y_pred = linear_regression.predict(X_test)

plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,linear_regression.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()