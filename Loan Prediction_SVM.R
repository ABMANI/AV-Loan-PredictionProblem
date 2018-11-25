install.packages("mice")
library(mice)
install.packages("mlr")
library(mlr)
setwd("E:/Analytics/Analytics Vidhya Datahack/24 Case Study Challenge/Loan Prediction")
train=read.csv("train_u6lujuX_CVtuZ9i.csv")
train[train==""]<-NA

normalize <- function(x)
{return((x- min(x)) /(max(x)-min(x)))}

#convert the text colums to numeric and then replace missing values with imputation using mice
train$Gender<-as.factor(as.numeric(train$Gender,na.rm = T)-2)
train$Married<-as.factor(as.numeric(train$Married,na.rm = T)-2)
train$Dependents<-as.factor(as.numeric(train$Dependents,na.rm = T)-1)
train$Education<-as.factor(as.numeric(train$Education,na.rm = T)-1)
train$Self_Employed<-as.factor(as.numeric(train$Self_Employed,na.rm = T)-1)
train$Property_Area<-as.factor(as.numeric(train$Property_Area,na.rm = T)-1)
train$Loan_Status<-as.factor(as.numeric(train$Loan_Status,na.rm = T)-1)
tr_loan_id<-train[,1]
train<-train[,-1]
loan_status<-train$Loan_Status
train<-train[,-12]
#create dummy variables for categorical columns
dmy<-dummyVars("~.",data=train[,])
train_tf<-data.frame(predict(dmy,newdata = train))
#missing value imputation
micemod<-mice(as.data.frame(train_tf),method = "rf")
miceoutput<-complete(micemod)
#summary(miceoutput)  
train_clean<-cbind(miceoutput,loan_status)  
train_clean<-train_clean[,-c(2,4,8,10,12,20)]
# To get a vector, use apply instead of lapply
train_clean$ApplicantIncome<-normalize(train_clean$ApplicantIncome)
train_clean$CoapplicantIncome<-normalize(train_clean$CoapplicantIncome)
train_clean$LoanAmount<-normalize(train_clean$LoanAmount)
train_clean$Loan_Amount_Term<-normalize(train_clean$Loan_Amount_Term)

#convert the test data for comparison
test=read.csv("test_Y3wMUE5_7gLdaTN.csv")
test[test==""]<-NA

test$Gender<-as.factor(as.numeric(test$Gender,na.rm = T)-2)
test$Married<-as.factor(as.numeric(test$Married,na.rm = T)-1)
test$Dependents<-as.factor(as.numeric(test$Dependents,na.rm = T)-1)
test$Education<-as.factor(as.numeric(test$Education,na.rm = T)-1)
test$Self_Employed<-as.factor(as.numeric(test$Self_Employed,na.rm = T)-1)
test$Property_Area<-as.factor(as.numeric(test$Property_Area,na.rm = T)-1)

test_id<-test[,1]
test<-test[,-1]

dmy<-dummyVars("~.",data=test[,])
test_tf<-data.frame(predict(dmy,newdata = test))
micemod_test<-mice(as.data.frame(test_tf),method = "rf")
miceoutput_test<-complete(micemod_test)
test_clean<-miceoutput_test[,-c(2,4,8,10,12,20)]
test_clean$loan_status<-sample(0:1,size=367,replace=T)
# To get a vector, use apply instead of lapply
test_clean$ApplicantIncome<-normalize(test_clean$ApplicantIncome)
test_clean$CoapplicantIncome<-normalize(test_clean$CoapplicantIncome)
test_clean$LoanAmount<-normalize(test_clean$LoanAmount)
test_clean$Loan_Amount_Term<-normalize(test_clean$Loan_Amount_Term)

# #break the train data into training and validation
# 
# partition<-round(0.75*nrow(train_clean),0)
# tr_clean<-train_clean[1:partition,]
# tr_cv_clean<-train_clean[461:614,]
####################################################################################################################################
#Trying SVM for binary classification

#create a task
train_task <- makeClassifTask(data = train_clean,target = "loan_status")
test_task <- makeClassifTask(data = test_clean,target = "loan_status")

#create learner
getParamSet("classif.ksvm")
bag <- makeLearner("classif.ksvm",predict.type = "response")


#Set parameters
pssvm <- makeParamSet(
  makeDiscreteParam("C", values = 2^c(-8,-4,-2,0)), #cost parameters
  makeDiscreteParam("sigma", values = 2^c(-8,-4,0,4)) #RBF Kernel Parameter
)

#specify search function
ctrl <- makeTuneControlGrid()

#set 3 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 3L)

#tune model
res <- tuneParams(bag, task = train_task, resampling = set_cv, par.set = pssvm, control = ctrl,measures = acc)

#CV accuracy
res$y
acc.test.mean 
0.8062171 

#set the model with best params
t.svm <- setHyperPars(bag, par.vals = res$x)

#train
par.svm <- train(bag, train_task)

#test
predict.svm <- predict(par.svm, test_task)
# #using hyperparameters for modeling
# rf.tree<-setHyperPars(rf.lrn,par.vals =tune$x )
# rforest<-train(rf.tree,train_task)
# 
# rfmodel <- predict(rforest, test_task)
predict.svm$data$response<-ifelse(predict.svm$data$response==1,"Y","N")



ss<-read.csv("Sample_Submission_ZAuTl8O_FK3zQHh.csv")
ss_1<-cbind(as.data.frame(test_id,stringasFactors=F),as.data.frame(predict.svm$data$response,stringasFactors=F))
colnames(ss_1)<-colnames(ss)
write.csv(ss_1,file = "Loan_Prediction_Submission_SVM.csv",row.names = F)