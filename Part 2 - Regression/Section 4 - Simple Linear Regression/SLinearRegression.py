# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 08:42:05 2019

@author: Yahya Ali
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 08:46:56 2019

@author: Yahya
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
DataSet = pd.read_csv('Salary_Data.csv')

X= pd.DataFrame(DataSet.iloc[:,0:1]).values
Y= pd.DataFrame(DataSet.iloc[:,1:2]).values

#For dividing into test and train test
from sklearn.model_selection import train_test_split

#test size is the ratio of split. 
#So that we can test if the ML model is good enough to predict or not
#we want a good split to prevent issues like overfitting 
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=1/3, random_state=0)


#import linear regressor that will fit to our data. 
#Linear regression is finding a best fit line between two variables

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor = regressor.fit(X_train,y_train)

#Next step is to understand how good it the model we have 
y_pred = regressor.predict(X_test)

#To have better understanding of the data we need to visualize them to see how good
#are the prediction
plt.scatter(X_train,y_train)
#We want to see the model that has been created by the regressor. Predict will 
#return values that 
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.scatter(X_test,y_pred,color="yellow")
plt.scatter(X_test,y_test,color="black")

plt.title("Exp vs Salary")
plt.xlabel("Years of Exp")
plt.ylabel("Salary")
plt.show()




