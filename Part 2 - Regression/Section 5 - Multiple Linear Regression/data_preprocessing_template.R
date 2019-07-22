# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('50_Startups.csv')

dataset$State = factor(dataset$State, levels= c('New York','Florida','California'),
                         labels = c(1,2,3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Linear Model for multiple variable 

lm = lm(formula = Profit ~ . ,data = training_set)


