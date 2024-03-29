#Data prepocessing 

#Import 
dataset = read.csv('Data.csv')


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