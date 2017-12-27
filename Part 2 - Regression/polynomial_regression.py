# Polynomial Regression
# Again, very similar to linear regression code wise;
# Uses PolynomialFeatures() to transform the matrix of features
# then LinearRegression() as normal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # Ensure that x is a matrix and not a vector (use 1:2 instead of just 1 for column)
y = dataset.iloc[:, 2].values

# No need to split the data into test and training, goal is to predict salary at level 6.5

# Feature Scaling is done automatically by PolynomialFeatures

# Fit a Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fit a Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# Transform the matrix of features
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

# Then proceed with linear regression as normal
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Comment out visualizations as necessary

# Visualize the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualize the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualize the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)     # for a smoother plot
X_grid = X_grid.reshape((len(X_grid), 1))   # reshape vector to matrix to plot 
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predict a new result with Linear Regression
lin_prediction = lin_reg.predict(6.5)

# Predict a new result with Polynomial Regression
poly_prediction = lin_reg_2.predict(poly_reg.fit_transform(6.5))

print('The proposed salary at level 6.5 is $160k\n The linear model predicts: ', lin_prediction[0])
print('\n The polynomial model predicts: ', poly_prediction[0])