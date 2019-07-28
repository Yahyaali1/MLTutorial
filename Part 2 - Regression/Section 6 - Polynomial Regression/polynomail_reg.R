# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

#creating linear model
#for comparison 

lin_reg = lm(formula= Salary ~ ., dataset)
summary(lin_reg)
#we can check the variables with summary that is contributing most
#in this model 

#add data level for additional polynomial features 
dataset$Level2 = dataset$Level^2
ply_reg = lm(formula = Salary ~ .,data = dataset )
summary(ply_reg)

#adding another polynomail feature of cofficient power 3 

dataset$Level3 = dataset$Level^3
ply_reg = lm(formula = Salary ~ .,data = dataset )
summary(ply_reg)

#Visualizing the results  

ggplot()+
  geom_point(aes(dataset$Level, dataset$Salary))+
  geom_line(aes(dataset$Level,predict(lin_reg,newdata = dataset)))+
  geom_line(aes(dataset$Level,predict(ply_reg,newdata = dataset)))
  

#Predicting one single value with linear model 

y_pred = predict(lin_reg, data.frame(Level = 6.5))

#predicting with polynomial model 

y_pred2 = predict(ply_reg, data.frame(Level = 6.5 ,
                                      Level2 = 6.5^2,
                                      Level3 = 6.5^3))

