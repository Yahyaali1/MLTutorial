# Summary 

### Data pre processing 
* Reading Data
* Test & Training Set Split
* Missing Data
* Categorical Data Handling
* Feature Scaling
- - - 
## Python

Reading data
```python
import pandas as pd
#Import Dataset
  DataSet = pd.read_csv('Data.csv')
  X= pd.DataFrame(DataSet.iloc[:,:-1]).values
```
Test & Train Split
```Python
#For dividing into test and train test
from sklearn.model_selection import train_test_split
#test size is the ratio of split. 
#So that we can test if the ML model is good enough to predict or not
#we want a good split to prevent issues like overfitting 
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
```
Missing Data
```Python
#solving issues of the missing data possible solution 
#1-Remove data : Not very good approach
#2- Take mean of the data 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean", axis=0)
imputer = imputer.fit(X[:,1:3]) #upper bound in excluded
X[:,1:3] = imputer.transform(X[:,1:3])

#For experimenting with mean and median 
median_imputer = Imputer(missing_values="NaN", strategy="median", axis=0)
```


Categorical Data Handling
```Python
#Moving categorical data into different columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
X_copy[:,0] = labelencoder_X.fit_transform(X_copy[:,0])

#To add new C for each country in this case for mathematical model, separate Column
#are needed as we don't want any country to have higer value in mathematical model 
one_hot_encoder = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder.fit_transform(X).toarray()
```

Feature Scaling 
```Python
from sklearn.preprocessing import StandardScaler
#Difference of opnion exist on having countries scaled as well. In this case we 
#are scaling them
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#Transfor is applied as the sc is already fit to train set
X_test = sc_X.transform(X_test)
```

- - -
