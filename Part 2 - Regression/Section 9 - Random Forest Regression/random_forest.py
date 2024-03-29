# Regression Template

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#importing the Decision Tree 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 100 , random_state=0)
regressor.fit(X,y)

#We can increase the number of the decision tree to predict better. 
#As more trees this means we are covering towards to the actual value. 

y_pred = regressor.predict(6.5)
# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()