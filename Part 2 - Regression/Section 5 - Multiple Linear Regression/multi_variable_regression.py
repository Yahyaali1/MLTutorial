# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#As we have categories for the 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3]) 

one_hot_encoder = OneHotEncoder(categorical_features=[3])
X = one_hot_encoder.fit_transform(X).toarray()

#Dummy variable trap
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression().fit(X_train,y_train)
y_pred= regressor.predict(X_test)

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((50,1), dtype=int), values = X , axis = 1)

regressor = backwardElimination(X,y,0.05)

#Function to find max index 
def maxValueIndex(arr,maxvalue):
	for j in range (0,len(arr)):
		if(arr[j]==maxvalue):
			return j
	return -1
#function to do automatic backward elimination 
def backwardElimination(X,y,SL):
	regressor = sm.OLS(endog = y, exog = X).fit()
	for i in range(0,np.size(X,axis=1)):
		if(np.amax(regressor.pvalues) > SL):
			index = maxValueIndex(regressor.pvalues,np.amax(regressor.pvalues))
			if(index!=-1):
				X = np.delete(X,index,1)
				regressor = sm.OLS(endog = y, exog = X).fit()
		
	return regressor
	
	
	
	
	
	
	
	
	