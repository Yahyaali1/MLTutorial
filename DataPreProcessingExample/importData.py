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
X= pd.DataFrame(DataSet.iloc[:,:-1].values)
Y= pd.DataFrame(DataSet.iloc[:,3].values)

#solving issues of the missing data 
#possible solution 
#1-Remove data : Not very good approach
#2- Take mean of the data 

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="nan",strategy="mean", axis=0)
imputer = imputer.fit(X[:,1:3]) #upper bound in excluded
X[:,1:3] = imputer.transform(X[:,1:3])
