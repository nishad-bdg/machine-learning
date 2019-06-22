# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)


from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train,y_train)

y_pred = regression.predict(X_test)

plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train,regression.predict(X_train), color = 'blue')
plt.title("Salary vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.plot()
