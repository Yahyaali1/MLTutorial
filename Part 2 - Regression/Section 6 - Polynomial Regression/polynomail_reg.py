# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#We will form two models for comparison
#Linear and Polynomial Model


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Increase the degree for better fittign graph 
ploy_fea = PolynomialFeatures(degree=2)
X_ploy = ploy_fea.fit_transform(X)

#We have used 2nd degree values to create a linear model 
lin_reg2 = LinearRegression()
lin_reg2.fit(X_ploy,y)

y_pred = lin_reg.predict(X)
plt.scatter(X,y, c="red")
plt.plot(X,y_pred, c="yellow")
plt.plot(X,lin_reg2.predict(X_ploy))

#Increase the degree for better fittign graph 
ploy_fea = PolynomialFeatures(degree=4)
X_ploy = ploy_fea.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_ploy,y)
#We are formaing another array to plot the graph for more interval to have smoother curve. 
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
X_grid_ploy = ploy_fea.fit_transform(X_grid)

y_pred = lin_reg.predict(X)
plt.plot(X,y_pred, c="yellow")
plt.scatter(X,y, c="red")
#we will use the new linear reg model for higer degree to perdict results for the data 
plt.plot(X_grid,lin_reg2.predict(X_grid_ploy))


