# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the dataset using pandas.
2. Select input features and target values, then normalize the data using StandardScaler.
3. Initialize theta values and train the Linear Regression model using Gradient Descent by updating weights iteratively.
4. Scale the new input data, predict the output using the trained model, and convert the predicted value back to the original scale.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: RAJARAMAN V
RegisterNumber:  212223110038
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros((X.shape[1], 1))
    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
    return theta

data = pd.read_csv("50_Startups.csv")

print("Dataset Preview:")
print(data.head())

X = data.iloc[:, :-2].values
y = data.iloc[:, -1].values.reshape(-1, 1)

X = X.astype(float)

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

theta = linear_regression(X_scaled, y_scaled)

print("\nTheta Values:")
print(theta)
new_data = np.array([[165349.2, 136897.8, 471784.1]])
new_scaled = X_scaler.transform(new_data)
new_scaled = np.c_[np.ones(len(new_scaled)), new_scaled]
prediction_scaled = new_scaled.dot(theta)
prediction = y_scaler.inverse_transform(prediction_scaled)

print("\nPredicted Profit:")
print(prediction)
```

## Output:
<img width="730" height="357" alt="image" src="https://github.com/user-attachments/assets/19df3d47-60c9-4e49-a864-19d08d41d1df" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
