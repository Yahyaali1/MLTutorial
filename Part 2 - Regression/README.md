# Summary 

*Requires understanding of the [datapreprocessing](https://github.com/Yahyaali1/MLTutorial/tree/master/DataPreProcessingExample)*
### Regression 
* Simple Linear Regression
* Multiple Linear Regression
* Polynomial Linear Regression
* Decision Tree Regression
* Random Forest Regression
* Evaluating Regression Model
- - - 
## Python

Simple Linear Regression
```python
#import linear regressor that will fit to our data. 
#Linear regression is finding a best fit line between two variables
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor = regressor.fit(X_train,y_train)

#To have better understanding of the data we need to visualize them to see how good
#are the prediction
plt.scatter(X_train,y_train)
```

- - -

## R

Simple Linear Regression
```R
#To form a linear model regressor 

regressor = lm(formula = Salary ~ YearsExperience, train_set)
#to check the summary of the model created 
summary(regressor)

#predict function is to take test set and predict the values out of it 
y_pred = predict(regressor,newdata = test_set)

```
