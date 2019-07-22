#Data prepocessing 

#Import 
dataset = read.csv('Salary_Data.csv')


#spliting the data into test and train set 
#installing the package required
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary,SplitRatio = 2/3)
train_set = subset(dataset,split==TRUE)
test_set = subset(dataset,split== FALSE)

#To form a linear model regressor 

regressor = lm(formula = Salary ~ YearsExperience, train_set)
#to check the summary of the model created 
summary(regressor)

#predict function is to take test set and predict the values out of it 
y_pred = predict(regressor,newdata = test_set)

#installing package for visulaizing the result of the data 
install.packages('ggplot2')

#adding test set as scatter plot and regression model as a straight line
ggplot()+
  geom_point(aes(train_set$YearsExperience, train_set$Salary))+
  geom_line(aes(train_set$YearsExperience,predict(regressor,newdata = train_set)))

#Checking the model with scatter plot of test, on how close the line passes by the origional data points 
ggplot()+
  geom_point(aes(test_set$YearsExperience, test_set$Salary))+
  geom_line(aes(train_set$YearsExperience,predict(regressor,newdata = train_set)))
