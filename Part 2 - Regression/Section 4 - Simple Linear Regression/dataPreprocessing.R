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