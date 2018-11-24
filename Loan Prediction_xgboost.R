install.packages("mice")
library(mice)
setwd("E:/Analytics/Analytics Vidhya Datahack/24 Case Study Challenge/Loan Prediction")
train=read.csv("train_u6lujuX_CVtuZ9i.csv")
train[train==""]<-NA
train_backup=read.csv("train_u6lujuX_CVtuZ9i.csv")
#summary(train)

#convert the text colums to numeric and then replace missing values with mode of values for categorical and mean for continuous variables
train$Gender<-as.factor(as.numeric(train$Gender,na.rm = T)-2)
train$Married<-as.factor(as.numeric(train$Married,na.rm = T)-2)
train$Dependents<-as.factor(as.numeric(train$Dependents,na.rm = T)-1)
train$Education<-as.factor(as.numeric(train$Education,na.rm = T)-1)
train$Self_Employed<-as.factor(as.numeric(train$Self_Employed,na.rm = T)-1)
train$Property_Area<-as.factor(as.numeric(train$Property_Area,na.rm = T)-1)
train$Loan_Status<-as.factor(as.numeric(train$Loan_Status,na.rm = T)-1)
train<-train[,-1]
loan_status<-train$Loan_Status
train<-train[,-12]

#create dummy variables for categorical columns
dmy<-dummyVars("~.",data=train[,])
train_tf<-data.frame(predict(dmy,newdata = train))

#missing value imputation
md.pattern(train)
micemod<-mice(as.data.frame(train_tf),method = "rf")
miceoutput<-complete(micemod)

#summary(miceoutput)  
train_clean<-cbind(miceoutput,loan_status)  
train_clean<-train_clean[,-c(2,4,8,10,12,20)]

#break the train data into training and validation

partition<-round(0.75*nrow(train_clean),0)
tr_clean<-train_clean[1:partition,]
tr_cv_clean<-train_clean[461:614,]

################################################################################################################################
#apply logisitic regression
#model<-glm(loan_status~.,data = tr_clean,family = binomial,method = "glm.fit")
#summary(model)
#anova(model,test="Chisq")

#predict <- predict(model,newdata = tr_cv_clean , type = 'response')
##################################################################################################################################

#apply xgboost using logistic regressionx
tr_labels<-tr_clean$loan_status
ts_labels<-tr_cv_clean$loan_status
new_tr<-model.matrix(~.+0,data = tr_clean[,-15]) 
new_ts<-model.matrix(~.+0,data = tr_cv_clean[,-15])


#convert factor to numeric 
tr_labels <- as.numeric(tr_labels)-1
ts_labels <- as.numeric(ts_labels)-1


#preparing matrix 
dtrain <- xgb.DMatrix(data = new_tr,label = tr_labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_labels)


#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", 
                 eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, 
                 showsd = T, stratified = T, print_every_n = 10, early_stop_round = 20, maximize = F)
min(xgbcv$test.error.mean)

xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 11, watchlist = list(val=dtest,train=dtrain), 
                   print_every_n = 10, early_stop_round = 10, maximize = F , eval_metric = "error")

xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)
####################################################################################################################################
#confusion matrix
table (xgbpred, ts_labels)
#table(tr_cv_clean$loan_status, predict > 0.5)

#ROCR Curve
library(ROCR)
ROCRpred <- prediction(predict, tr_cv_clean$loan_status)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))
#plot glm
library(ggplot2)
ggplot(tr_cv_clean, aes(x=loan_status, y=predict)) + geom_point() + 
  stat_smooth(method="glm", family="binomial", se=FALSE)

#convert the test data for comparison
test=read.csv("test_Y3wMUE5_7gLdaTN.csv")
test[test==""]<-NA

test$Gender<-as.factor(as.numeric(test$Gender,na.rm = T)-2)
test$Married<-as.factor(as.numeric(test$Married,na.rm = T)-1)
test$Dependents<-as.factor(as.numeric(test$Dependents,na.rm = T)-1)
test$Education<-as.factor(as.numeric(test$Education,na.rm = T)-1)
test$Self_Employed<-as.factor(as.numeric(test$Self_Employed,na.rm = T)-1)
test$Property_Area<-as.factor(as.numeric(test$Property_Area,na.rm = T)-1)
#train$Loan_Status<-as.factor(as.numeric(train$Loan_Status,na.rm = T)-1)
test_id<-test[,1]
test<-test[,-1]
#loan_status<-train$Loan_Status
#train<-train[,-12]
dmy<-dummyVars("~.",data=test[,])
test_tf<-data.frame(predict(dmy,newdata = test))
micemod_test<-mice(as.data.frame(test_tf),method = "rf")
miceoutput_test<-complete(micemod_test)
#test_clean<-cbind(miceoutput_test,loan_status)  
test_clean<-miceoutput_test[,-c(2,4,8,10,12,20)]

#test_predict<-predict(model,newdata=test_clean,type = 'response')
newer_ts<-model.matrix(~.+0,data = test_clean)
mtest <- xgb.DMatrix(data = newer_ts)
test_predict<-predict(xgb1,mtest)
test_loan_status<-factor(test_predict>0.5,labels = c("N","Y"))
ss<-read.csv("Sample_Submission_ZAuTl8O_FK3zQHh.csv")
ss_1<-cbind(as.data.frame(test_id,stringasFactors=F),as.data.frame(test_loan_status,stringasFactors=F))
colnames(ss_1)<-colnames(ss)
write.csv(ss_1,file = "Loan_Prediction_Submission_XGBoost.csv")