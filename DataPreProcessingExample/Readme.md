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

## R

Reading data
```R
#Import 
dataset = read.csv('Data.csv')
```

Test & Train Split
```R
#spliting the data into test and train set 
#installing the package required
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio = 0.8)
train_set = subset(dataset,split==TRUE)
test_set = subset(dataset,split== FALSE)
```
Missing Data
```R
#solving issues of the missing data possible solution 
#1-Remove data : Not very good approach
#2- Take mean of the data 
#Adding Missing Values 
#Checks if the data is not available in the column. Uses function to add mean value.
dataset$Age = ifelse(is.na(dataset$Age),ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),dataset$Salary)
```


Categorical Data Handling
```R
#Encoding the country 
#R uses factors instead of encoding the data into multiple columns. We need to specify labels for
#each encoding level
dataset$Country = factor(dataset$Country, levels= c('France','Germany','Spain'),
                         labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased, levels= c('Yes','No'),
                         labels = c(1,0))
```

Feature Scaling 
```R
#scale
#we have to specify columns as the factors are not numeric values
#Upper 
train_set[,2:3] = scale(train_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])
```
