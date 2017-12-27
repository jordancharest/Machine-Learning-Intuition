# Multiple Linear Regression
# Code wise, nearly the same as simple linear regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoid the Dummy Variable Trap
# sklearn LinearRegression() does this automatically
X = X[:, 1:]

# Split the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling is done automatically by sklearn.LinearRegression()

# Fit Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Optimize the model using Backward Elimination
import statsmodels.formula.api as sm
# statsmodels library does not include a constant factor in the linear regression
# equation, so we must include it (as a column of ones in the matrix of features)
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_optimal = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()   # OLS = Ordinary Least Squares

# Output useful statistical information about our model
regressor_OLS.summary()

# Using significance level of 5%, remove 'Florida' as an independent variable, repeat process
X_optimal = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

# Using significance level of 5%, remove 'New York', repeat process
X_optimal = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

# Remove 'Administration spend', repeat process
X_optimal = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()

# Reattempt the Linear Regression, removing the statistically insignificant variables
X_train, X_test, y_train, y_test = train_test_split(X_optimal, y, test_size = 0.2, random_state = 0)
regressor.fit(X_train, y_train)
y_pred_optimized = regressor.predict(X_test)

