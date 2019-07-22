#Data prepocessing 

#Import 
dataset = read.csv('Data.csv')

#Adding Missing Values 
dataset$Age = ifelse(is.na(dataset$Age),ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),dataset$Age)


dataset$Salary = ifelse(is.na(dataset$Salary),ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),dataset$Salary)

#Encoding the country 

dataset$Country = factor(dataset$Country, levels= c('France','Germany','Spain'),
                         labels = c(1,2,3))


dataset$Purchased = factor(dataset$Purchased, levels= c('Yes','No'),
                         labels = c(1,0))

#spliting the data into test and train set 
#installing the package required
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio = 0.8)
train_set = subset(dataset,split==TRUE)
test_set = subset(dataset,split== FALSE)

#scale
#we have to specify columns as the factors are not numeric values
train_set[,2:3] = scale(train_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])