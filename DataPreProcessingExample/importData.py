# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 08:46:56 2019

@author: Yahya
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
DataSet = pd.read_csv('Data.csv')

X= pd.DataFrame(DataSet.iloc[:,:-1]).values
X_copy= pd.DataFrame(DataSet.iloc[:,:-1]).values
Y= pd.DataFrame(DataSet.iloc[:,3]).values

#solving issues of the missing data 
#possible solution 
#1-Remove data : Not very good approach
#2- Take mean of the data 

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN",strategy="mean", axis=0)
imputer = imputer.fit(X[:,1:3]) #upper bound in excluded
X[:,1:3] = imputer.transform(X[:,1:3])

#For experimenting with mean and median 
median_imputer = Imputer(missing_values="NaN", strategy="median", axis=0)

median_imputer = median_imputer.fit(X_copy[:,1:3]) #upper bound in excluded
X_copy[:,1:3] = median_imputer.transform(X_copy[:,1:3])

#Moving categorical data into different columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
X_copy[:,0] = labelencoder_X.fit_transform(X_copy[:,0])

#To add new C for each country in this case for mathematical model, separate Column
#are needed as we don't want any country to have higer value in mathematical model 
one_hot_encoder = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder.fit_transform(X).toarray()
X_copy = one_hot_encoder.fit_transform(X_copy).toarray()

labelencoder_Y = LabelEncoder()
Y[:,0]=labelencoder_Y.fit_transform(Y[:,0])

#For dividing into test and train test
from sklearn.model_selection import train_test_split

#test size is the ratio of split. 
#So that we can test if the ML model is good enough to predict or not
#we want a good split to prevent issues like overfitting 
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
X_copy_train,X_copy_test,y_copy_train,y_copy_test = train_test_split(X,Y,test_size=0.2, random_state=0)

#As we can observe that data is not normalized, it can effec our ML model
#We have two options to normalize data
#1- Standardistaion(by std and mean) 2-Normalisation (by min and max value)
from sklearn.preprocessing import StandardScaler

#Difference of opnion exist on having countries scaled as well. In this case we 
#are scaling them
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#Transfor is applied as the sc is already fit to train set
X_test = sc_X.transform(X_test)

sc_X_copy = StandardScaler()
X_copy_train = sc_X.fit_transform(X_train)
X_copy_test = sc_X.transform(X_test)





