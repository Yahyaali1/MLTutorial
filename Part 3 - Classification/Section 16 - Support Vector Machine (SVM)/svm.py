# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas_ml import ConfusionMatrix

def plotConfusionMatrix(confusionMatrix,x_labels):
	ax = plt.subplot()
	sns.heatmap(confusionMatrix, annot=True, ax = ax)
	ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
	ax.set_title('Confusion Matrix'); 
	ax.xaxis.set_ticklabels(x_labels);
	ax.yaxis.set_ticklabels(x_labels[::-1])
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

# Fitting SVM classifier on the data set
from sklearn.svm import SVC
#Using linear kernal 
#Using linear kernal gives us linear or logistic results
#Another approach is to use polynomial kernal and add respective degree
#For linear case we can use degree=1 
classifier_linear = SVC(kernel="poly",random_state=0,degree=1)

classifier_linear.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier_linear.predict(X_test)


cm_linear = ConfusionMatrix(y_test, y_pred)
cm_linear

#Changing the classifier to achieve better results
classifier_rbf = SVC(kernel="rbf",random_state=0)
classifier_rbf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier_rbf.predict(X_test)

cm_pandas = ConfusionMatrix(y_test,y_pred)
cm_pandas


X_set = X_train
y_set = y_train
classifier=classifier_linear
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


	