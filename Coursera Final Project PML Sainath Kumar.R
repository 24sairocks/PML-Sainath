library(class)
library(NLP)
library(tm)
library(rastor)
library(svmpath)
library(RTextTools)
library(klaR)
library(e1071)
library(plyr)
library(kernlab)
library(caret)
library(ISLR)
library(ggplot2)
library(Hmisc)
library(splines)
library(rpart)
#library(rattle)
library(ElemStatLearn)
library(party)
library(AppliedPredictiveModeling)
library(pgmm)
library(quantmod)

#Load the training and testing data sets
setwd("C:/Users/IBM_ADMIN/Documents/Coursera/Practical Machine LEarning")
train <-read.csv("pml-training.csv",header=TRUE)
test <-read.csv("pml-testing.csv",header=TRUE)
dim(train)
#Analyze the data set


#Based on initial look at the data the data has blanks , NA ,#DIV/0
#First step would be to cleanse the data
#Also at first glance the first 7 columns do not seem like the ones that would have effect on the prediction outcome .

train <-read.csv("pml-training.csv",header=TRUE,na.strings=c("NA","#DIV/0!",""))
test <-read.csv("pml-testing.csv",header=TRUE,na.strings=c("NA","#DIV/0!",""))
train <- train[,8:160]

#Identifying and removing near zero variance variables

train_var <- nearZeroVar(train, saveMetrics=TRUE)
train_var1 <- train[,train_var$nzv==FALSE]


train_1 <- train_var1
for(i in 1:length(train_var1)) {
  if( sum( is.na( train_var1[, i] ) ) /nrow(train_var1) >= .6) {
    for(j in 1:length(train_1)) {
      if( length( grep(names(train_var1[i]), names(train_1)[j]) ) == 1)  {
        train_1 <- train_1[ , -j]
      }
    }
  }
}

# Set back to the original variable name
train_var1 <- train_1
rm(train_1)
dim(train_var1)

#Applying the same transformations to the testing set as well
#Since here we have removed the near zero variables and NA variables testing data should also have only these variables

train_variables <- colnames(train_var1)
train_variables_1 <- colnames(train_var1[, -53])  # remove the classe column
test <- test[train_variables_1]
dim(test)


#Partitioning the training data
intrain <- createDataPartition(train_var1$classe, p=0.7, list=FALSE)
train_fin <- train_var1[intrain, ]
valid_fin <- train_var1[-intrain, ]
dim(train_fin)
dim(valid_fin)

#Using basic classification tree algorithm to predict the classe

modfit <- train(train_fin$classe ~ ., data = train_fin, method="rpart")
print(modfit)

#Running the Model against the test class

predict_1 <- predict(modfit, newdata=valid_fin)
print(confusionMatrix(predict_1,valid_fin$classe))

# Confusion Matrix and Statistics
#
# Reference
# Prediction    A    B    C    D    E
# A 1522  484  482  442  145
# B   21  389   39  164  140
# C  127  266  505  358  293
# D    0    0    0    0    0
# E    4    0    0    0  504
#
# Overall Statistics
#
# Accuracy : 0.4962
# 95% CI : (0.4833, 0.509)
# No Information Rate : 0.2845
# P-Value [Acc > NIR] : < 2.2e-16
#
# Kappa : 0.3413
# Mcnemar's Test P-Value : NA
#
# Statistics by Class:
#
# Class: A Class: B Class: C Class: D Class: E
# Sensitivity            0.9092   0.3415  0.49220   0.0000  0.46580
# Specificity            0.6312   0.9233  0.78514   1.0000  0.99917
# Pos Pred Value         0.4950   0.5166  0.32602      NaN  0.99213
# Neg Pred Value         0.9459   0.8539  0.87984   0.8362  0.89251
# Prevalence             0.2845   0.1935  0.17434   0.1638  0.18386
# Detection Rate         0.2586   0.0661  0.08581   0.0000  0.08564
# Detection Prevalence   0.5225   0.1280  0.26321   0.0000  0.08632
# Balanced Accuracy      0.7702   0.6324  0.63867   0.5000  0.73249





#the accuracy of the prediction deteriorated further less that 50% therefore this would not be a good model to predict
#Do a check if preprocess ing the data will improve the prediction results
modfit_1 <- train(train_fin$classe ~ ., data = train_fin, method="rpart", preProcess=c("center", "scale"))

print(modfit_1)

#Running the Model against the test class

predict_1 <- predict(modfit_1, newdata=valid_fin)
print(confusionMatrix(predict_1,valid_fin$classe))

#Preprocessing does not improve the accruacy further therfore proceeding to use other algrithms to build the model
#Random forests and Boosting algorithms to be used to build further models

modfit_2 <- train(train_fin$classe ~ ., data = train_fin, method="rf", preProcess=c("center", "scale"))

print(modfit_2)

#Running the Model against the test class

predict_2 <- predict(modfit_2, newdata=valid_fin)
print(confusionMatrix(predict_2,valid_fin$classe))


#
# > modfit_2 <- train(train_fin$classe ~ ., data = train_fin, method="rf", preProcess=c("center", "scale"))
# >
#   > print(modfit_2)
# Random Forest
#
# 13737 samples
# 52 predictor
# 5 classes: 'A', 'B', 'C', 'D', 'E'
#
# Pre-processing: centered (52), scaled (52)
# Resampling: Bootstrapped (25 reps)
# Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ...
# Resampling results across tuning parameters:
#
#   mtry  Accuracy   Kappa
# 2    0.9881682  0.9850310
# 27    0.9887566  0.9857768
# 52    0.9811102  0.9761049
#
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was mtry = 27.
# >
#   > #Running the Model against the test class
#   >
#   > predict_2 <- predict(modfit_2, newdata=valid_fin)
# > print(confusionMatrix(predict_2,valid_fin$classe))
# Confusion Matrix and Statistics
#
# Reference
# Prediction    A    B    C    D    E
# A 1674   15    0    0    0
# B    0 1123    4    0    0
# C    0    1 1017    9    0
# D    0    0    5  955    0
# E    0    0    0    0 1082
#
# Overall Statistics
#
# Accuracy : 0.9942
# 95% CI : (0.9919, 0.996)
# No Information Rate : 0.2845
# P-Value [Acc > NIR] : < 2.2e-16
#
# Kappa : 0.9927
# Mcnemar's Test P-Value : NA
#
# Statistics by Class:
#
# Class: A Class: B Class: C Class: D Class: E
# Sensitivity            1.0000   0.9860   0.9912   0.9907   1.0000
# Specificity            0.9964   0.9992   0.9979   0.9990   1.0000
# Pos Pred Value         0.9911   0.9965   0.9903   0.9948   1.0000
# Neg Pred Value         1.0000   0.9966   0.9981   0.9982   1.0000
# Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
# Detection Rate         0.2845   0.1908   0.1728   0.1623   0.1839
# Detection Prevalence   0.2870   0.1915   0.1745   0.1631   0.1839
# Balanced Accuracy      0.9982   0.9926   0.9946   0.9948   1.0000

#Plotting the importance of the variables/predictors
print(plot(varImp(modfit_2, scale = FALSE)))


#Applying the model to the final test data for 20 cases
# Run against 20 testing set provided by Professor Leek.
print(predict(modfit_2, newdata=test))
# [1] B A B A A E D B A A B C B A E E A B B B
# Levels: A B C D E

Results <- data.frame(predicted=predict(modfit_2, newdata=test)
)
print(Results)
#Conclusion:
#
#Based on the RF model the test cases have been predicted as above.The accuracy for the RF model is relatively higher and would certainly provide more accurate results
#Additional Analysis and further explanation:
#I would like to have done additional exploratory data analysis to reduce the number of predictors down further. PCA- Principal component analysis as well as using concepts from Regularized regression and combining predictor chapter.

