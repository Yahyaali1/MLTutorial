# Regression Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]
#Installing the package for the decision tree 
install.packages('rpart')
#including the library 
library('rpart')


#building the decision tree regressor

regressor = rpart(formula = dataset$Salary ~ .,dataset)

# Predicting a new result
#The results are not good as the decision tree is making only one split :( 

y_pred = predict(regressor, data.frame(Level = 6.5))


#we will use rpart to add min number of splits there should be 

regressor = rpart(formula = dataset$Salary ~ .,dataset,
                  control = rpart.control(minsplit = 5))

# Predicting a new result
#The results are not good as the decision tree is making only one split :( 

y_pred = predict(regressor, data.frame(Level = 6.5))



# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Descison Regression Model)') +
  xlab('Level') +
  ylab('Salary')