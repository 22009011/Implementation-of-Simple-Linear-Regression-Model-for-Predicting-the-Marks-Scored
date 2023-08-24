# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: THANJIYAPPAN.K
RegisterNumber: 212222240108

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
x = dataset.iloc[:,:-1].values
print(x)
y = dataset.iloc[:,-1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color="orange")
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
### dataset head()
![image](https://github.com/22009011/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343461/2ff58d1e-9072-4a14-8a65-c989e69cc7ab)
### dataset tail()
![image](https://github.com/22009011/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343461/e57ce56b-0805-406b-8d36-02bad5bfebad)
#Array value of X


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
