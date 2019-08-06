# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

X_set = X_train
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
plt.scatter(X_train[:,0][y_train==0],X_train[:,1][y_train==0], marker = ".")
plt.scatter(X_train[:,0][y_train==1],X_train[:,1][y_train==1], marker = "+")

plt.show()