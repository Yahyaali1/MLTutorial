
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas_ml import ConfusionMatrix

	
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling is not required in the case of decision tree. However 
# we need to perform this action to draw graph at certain for certain resolution 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#We can use several criterion for bases.
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy",random_state=0)

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


cm = ConfusionMatrix(y_test, y_pred)
cm




X_set = X_test
y_set = y_test
classifier=classifier
# Visualising the Training set results
#Visualize the results using scatter plot 
#Forming the under laying data
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#Ravel flatens the array 
#NpArray helps form array 
#T is for transpose 
#Classifier predicts results against all the possible outputs 
y_pred_test = classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)
#Seperating True Values for pixel values 
X1_true = X1.ravel()[y_pred_test==1] 
X2_true = X2.ravel()[y_pred_test==1]
#Separating false values for pixel values 
X1_false = X1.ravel()[y_pred_test==0]
X2_false = X2.ravel()[y_pred_test==0]
#Plotting Pixel points 
plt.scatter(X1_true,X2_true,c="green", alpha=0.1)
plt.scatter(X1_false,X2_false,c="red", alpha=0.3)
#PLotting true data values 
plt.scatter(X_set[:,0][y_set==0],X_set[:,1][y_set==0], marker = ".")
plt.scatter(X_set[:,0][y_set==1],X_set[:,1][y_set==1], marker = "+")

plt.show()


	