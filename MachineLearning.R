setwd("D:/R/R-travail/MOOC R programming march2015/8-Machine Learning")
getwd()

library(caret)
library(kernlab)
data(spam)
inTrain <- createDataPartition(y=spam$type, p=0.75, list=FALSE) # to create a training and test dataset
training <- spam[inTrain,]
testing <- spam[-inTrain,]
dim(training)
# K-folds
set.seed(32323)
folds <- createFolds(y=spam$type, k=10, list=TRUE, returnTrain=TRUE)
sapply(folds,length)
folds[[1]][1:10]
# to return the test set
set.seed(32323)
folds <- createFolds(y=spam$type, k=10, list=TRUE, returnTrain=FALSE)
sapply(folds,length)
folds[[1]][1:10]
# resampling
set.seed(32323)
folds <- createResample(y=spam$type, times=10, list=TRUE)
sapply(folds,length)
folds[[1]][1:10]
# time slices
set.seed(32323)
tme <- 1:1000 # time vector
folds <- createTimeSlices(y=tme, initialWindow=20, horizon=10) # 20 samples per window, horizon=10 is to predict the next 10 samples after the initial window
names(folds)
folds$train[[1]]
folds$test[[1]]

# training options
library(caret)
library(kernlab)
data(spam)
inTrain <- createDataPartition(y=spam$type, p=0.75, list=FALSE) # to create a training and test dataset
training <- spam[inTrain,]
testing <- spam[-inTrain,]
modelFit <- train(type~.,data=training, method="glm")
args(train.default)
args(trainControl)

# Plotting predictors
library(ISLR)
library(ggplot2)
library(caret)
data(Wage)
summary(Wage)
# get training/test sets
inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
dim(training); dim(testing)
featurePlot(x=training[,c("age","education","jobclass")], y=training$wage, plot="pairs")
qplot(age,wage,data=training)
qplot(age,wage,colour=jobclass,data=training)
qq <- qplot(age,wage,colour=education,data=training)
qq + geom_smooth(method="lm", formula=y~x)
library(Hmisc)
cutWage <- cut2(training$wage, g=3)
table(cutWage)
p1 <- qplot(cutWage, age, data=training, fill=cutWage, geom=c("boxplot"))
p1
p2 <- qplot(cutWage, age, data=training, fill=cutWage, geom=c("boxplot","jitter"))
library(gridExtra)
grid.arrange(p1,p2,ncol=2)
t1 <- table(cutWage, training$jobclass)
t1
prop.table(t1,1)
qplot(wage, colour=education, data=training, geom="density")

# basic preprocessing
library(caret)
library(kernlab)
data(spam)
inTrain <- createDataPartition(y=spam$type, p=0.75, list=FALSE) # to create a training and test dataset
training <- spam[inTrain,]
testing <- spam[-inTrain,]
hist(training$capitalAve, main="", xlab="ave. capital run length")
mean(training$capitalAve)
sd(training$capitalAve)
trainCapAve <- training$capitalAve
trainCapAveS <- (trainCapAve - mean(trainCapAve))/sd(trainCapAve)
mean(trainCapAveS)
sd(trainCapAveS)
# on the test set:
testCapAve <- testing$capitalAve
testCapAveS <- (testCapAve - mean(trainCapAve))/sd(trainCapAve)
mean(testCapAveS)
sd(testCapAveS)
# with preProcess:
preObj <- preProcess(training[,-58], method=c("center","scale"))
trainCapAveS <- predict(preObj, training[,-58])$capitalAve
mean(trainCapAveS)
sd(trainCapAveS)
# on the test set:
testCapAveS <- predict(preObj, testing[,-58])$capitalAve
mean(testCapAveS)
sd(testCapAveS)
# apply in train directly:
set.seed(32343)
modelFit <- train(type ~., data=training, preProcess=c("center","scale"), method="glm")
modelFit
# boxcox transformation
preObj <- preProcess(training[,-58], method=c("BoxCox"))
trainCapAveS <- predict(preObj, training[,-58])$capitalAve
par(mfrow=c(1,2))
hist(trainCapAveS)
qqnorm(trainCapAveS)
# NA
set.seed(13343)
# make some values NA
training$capAve <- training$capitalAve
selectNA <- rbinom(dim(training)[1], size=1, prob=0.05)==1
training$capAve[selectNA] <- NA
# impute and standardize
preObj <- preProcess(training[,-58], method="knnImpute")
capAve <- predict(preObj, training[,-58])$capAve
# standardize true values
capAveTruth <- training$capitalAve
capAveTruth <- (capAveTruth-mean(capAveTruth))/sd(capAveTruth)
# to see the difference between real and umpute values:
quantile(capAve - capAveTruth)
quantile((capAve-capAveTruth)[selectNA])
quantile((capAve-capAveTruth)[!selectNA])

# covariate creation
library(ISLR)
library(caret)
data(Wage)
inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]

table(training$jobclass)
dummies <- dummyVars(wage ~ jobclass, data=training)
head(predict(dummies, newdata=training))

nsv <- nearZeroVar(training, saveMetrics=TRUE)
nsv

library(splines)
bsBasis <- bs(training$age, df=3) # 3rd degree polynomial
head(bsBasis)
lm1 <- lm(wage~bsBasis, data=training)
par(mfrow=c(1,1))
plot(training$age, training$wage, pch=19, cex=0.5)
points(training$age, predict(lm1, newdata=training), col="red", pch=19, cex=0.5)
head(predict(bsBasis, age=testing$age))

# PCA preProcess
library(caret)
library(kernlab)
data(spam)
inTrain <- createDataPartition(y=spam$type, p=0.75, list=FALSE) # to create a training and test dataset
training <- spam[inTrain,]
testing <- spam[-inTrain,]
M <- abs(cor(training[,-58])) # 58= spam/ham
diag(M) <- 0 # diag=cor(x1 by x1)=1 so not interesting. So put = 0
which(M > 0.8, arr.ind=T)
names(spam)[c(34,32)]
plot(spam[,34], spam[,32])
X <- 0.71*training$num415 + 0.71*training$num857
Y <- 0.71*training$num415 - 0.71*training$num857
plot(X,Y)

smallSpam <- spam[,c(34,32)] # just num857 and num415
prComp <- prcomp(smallSpam)
plot(prComp$x[,1],prComp$x[,2])
prComp$rotation

typeColor <- ((spam$type=="spam")*1+1) # give 1 or 2, to have black or red
prComp <- prcomp(log10(spam[,-58]+1))
plot(prComp$x[,1], prComp$x[,2], col=typeColor,xlab="PC1",ylab="PC2")

# PCA in caret
preProc <- preProcess(log10(spam[,-58]+1), method="pca", pcaComp=2)
spamPC <- predict(preProc, log10(spam[,-58]+1))
plot(spamPC[,1], spamPC[,2], col=typeColor)

# on training set
preProc <- preProcess(log10(training[,-58]+1), method="pca", pcaComp=2)
trainPC <- predict(preProc, log10(training[,-58]+1))
modelFit <- train(training$type ~ ., method="glm", data=trainPC)
testPC <- predict(preProc, log10(testing[,-58]+1))
confusionMatrix(testing$type, predict(modelFit, testPC))

# alternative
modelFit <- train(training$type ~ ., method="glm", preProcess="pca", data=training)
confusionMatrix(testing$type, predict(modelFit, testing))

# predicting with regression
library(caret)
data(faithful)
set.seed(333)
inTrain <- createDataPartition(y=faithful$waiting, p=0.5, list=FALSE)
trainFaith <- faithful[inTrain,]
testFaith <- faithful[-inTrain,]
head(trainFaith)
plot(trainFaith$waiting, trainFaith$eruptions, pch=19, col="blue", xlab="Waiting",ylab="Duration")
lm1 <- lm(eruptions ~ waiting, data=trainFaith)
summary(lm1)
lines(trainFaith$waiting, lm1$fitted, lwd=3)

coef(lm1)[1] + coef(lm1)[2]*80
newdata <- data.frame(waiting=80)
predict(lm1, newdata)

par(mfrow=c(1,2))
plot(trainFaith$waiting, trainFaith$eruptions, pch=19, col="blue", xlab="Waiting",ylab="Duration")
lines(trainFaith$waiting, lm1$fitted, lwd=3)
plot(testFaith$waiting, testFaith$eruptions, pch=19, col="green", xlab="Waiting",ylab="Duration")
lines(testFaith$waiting, predict(lm1, newdata=testFaith), lwd=3)
par(mfrow=c(1,1))

sqrt(sum((lm1$fitted-trainFaith$eruptions)^2)) # calculate RMSE on training
sqrt(sum((predict(lm1,newdata=testFaith)-testFaith$eruptions)^2)) # calculate RMSE on test

pred1 <- predict(lm1, newdata=testFaith, interval="prediction") # we want an interval
ord <- order(testFaith$waiting) # ordering the values of test set
plot(testFaith$waiting, testFaith$eruptions, pch=19, col="blue")
matlines(testFaith$waiting[ord],pred1[ord,],type="l",col=c(1,2,2),lty=c(1,1,1),lwd=3)

modFit <- train(eruptions ~waiting, data=trainFaith, method="lm")
summary(modFit$finalModel)

# Predicting with regression Multiple Covariates
library(ISLR)
library(ggplot2)
library(caret)
data(Wage)
Wage <- subset(Wage, select=-c(logwage))
summary(Wage)

inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
dim(training)
dim(testing)

featurePlot(x=training[,c("age","education","jobclass")],y=training$wage, plot="pairs")
qplot(age,wage,data=training)
qplot(age,wage,colour=jobclass, data=training)
qplot(age,wage,colour=education, data=training)

modFit <- train(wage~age+jobclass+education, method="lm", data=training)
finMod <- modFit$finalModel
print(modFit)
plot(finMod,1,pch=19,cex=0.5,col="#00000010")
qplot(finMod$fitted, finMod$residuals,colour=race,data=training)
plot(finMod$residuals,pch=19)
pred <- predict(modFit, testing)
qplot(wage,pred,colour=year,data=testing)

modFitAll <- train(wage~., method="lm", data=training)
pred <- predict(modFitAll, testing)
qplot(wage,pred,data=testing)

#-----------QUIZZZZZZ ----------------------
library(AppliedPredictiveModeling)
library(caret)
d=data(AlzheimerDisease)
names(d)

library(AppliedPredictiveModeling)
data(concrete)
head(mixtures)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]     
summary(concrete)

library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
data <- training[,grep("^IL", names(training))]
names(data)
preProc <- preProcess(data, method="pca", thresh=0.91)

names(data2)
data2 <- cbind(training$diagnosis,data)
names(data2)[1] <- "diagnosis"
model1 <- train(diagnosis ~ ., method="glm", data=data2)
confusionMatrix(testing$diagnosis, predict(model1, testing))
# Acc modelnon PCA = 0.6463
preProc <- preProcess(data2[,-1], method="pca", thresh=0.81)
trainPC <- predict(preProc, data2[,-1])
model2 <- train(diagnosis~., method="glm", data=trainPC)
datatest <- testing[,grep("^IL", names(testing))]
datatest <- cbind(testing$diagnosis,datatest)
names(datatest)[1] <- "diagnosis"
testPC <- predict(preProc, datatest[,-1])
confusionMatrix(testing$diagnosis, predict(model2, testPC))
names(testing)

data(iris)
library(ggplot2)
names(iris)
table(iris$Species)
# training / test sets
inTrain <- createDataPartition(y=iris$Species, p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training); dim(testing)

qplot(Petal.Width, Sepal.Width, colour=Species, data=training)

library(rpart)
modFit <- train(Species ~ ., method="rpart", data=training)
print(modFit$finalModel)
par(mar=c(2,2,2,2))
plot(modFit$finalModel, uniform=TRUE, main="Classification Tree")
text(modFit$finalModel, use.n=TRUE, all=TRUE, cex=0.5)

library(rattle)
fancyRpartPlot(modFit$finalModel)

predict(modFit, newdata=testing)

# Bagging
library(ElemStatLearn)
data(ozone)
ozone <- ozone[order(ozone$ozone),]
head(ozone)

ll <- matrix(NA, nrow=10, ncol=155)
for(i in 1:10){
     ss <- sample(1:dim(ozone)[1],replace=TRUE)
     ozone0 <- ozone[ss,]
     ozone0 <- ozone0[order(ozone0$ozone),]
     loess0 <- loess(temperature ~ ozone, data=ozone0, span=0.2)
     ll[i,] <- predict(loess0, newdata=data.frame(ozone=1:155))
}

plot(ozone$ozone, ozone$temperature, pch=19, cex=0.5)
for(i in 1:10){
     lines(1:155, ll[i,],col="grey", lwd=2)
}
lines(1:155, apply(ll,2,mean),col="red",lwd=2)

predictors <- data.frame(ozone=ozone$ozone)
temperature <- ozone$temperature
treebag <- bag(predictors, temperature, B=10, bagControl = bagControl(fit=ctreeBag$fit, predict=ctreeBag$pred, aggregate=ctreeBag$aggregate))
plot(ozone$ozone, temperature, col="lightgrey", pch=19)
points(ozone$ozone, predict(treebag$fits[[1]]$fit,predictors),pch=19,col="red")
points(ozone$ozone, predict(treebag,predictors),pch=19,col="blue")

ctreeBag$fit
ctreeBag$pred
ctreeBag$aggregate

# random forests
data(iris)
library(ggplot2)
inTrain <- createDataPartition(y=iris$Species, p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]

modFit <- train(Species~., data=training,method="rf",prox=TRUE)
modFit
getTree(modFit$finalModel, k=2)

irisP <- classCenter(training[,c(3,4)], training$Species, modFit$finalModel$prox)
irisP <- as.data.frame(irisP)
irisP$Species <- rownames(irisP)
p <- qplot(Petal.Width, Petal.Length, col=Species, data=training)
p + geom_point(aes(x=Petal.Width, y=Petal.Length, col=Species),size=5,shape=4,data=irisP)

pred <- predict(modFit, testing)
testing$predRight <- pred==testing$Species
table(pred,testing$Species)
qplot(Petal.Width, Petal.Length, colour=predRight, data=testing, main="newdata predictions")

# boosting
library(ISLR)
data(Wage)
library(ggplot2)
library(caret)
Wage <- subset(Wage, select=-c(logwage))
inTrain <- createDataPartition(y=Wage$wage,p=0.7,list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]
modFit <- train(wage~., method="gbm",data=training,verbose=FALSE)
print(modFit)
qplot(predict(modFit,testing),wage,data=testing)

# model based prediction
data(iris)
library(ggplot2)
names(iris)
table(iris$Species)

inTrain <- createDataPartition(y=iris$Species, p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]

modlda <- train(Species ~ ., data=training, method="lda")
modnb <- train(Species ~ ., data=training, method="nb")
plda <- predict(modlda,testing)
pnb <- predict(modnb, testing)
table(plda,pnb)

equalPredictions <- (plda==pnb)
qplot(Petal.Width, Sepal.Width, colour=equalPredictions, data=testing)

# ----------------- QUIZZZZ 3 -------------------
library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)
?segmentationOriginal
training <- subset(segmentationOriginal, segmentationOriginal$Case == "Train")
testing <- subset(segmentationOriginal, segmentationOriginal$Case == "Test")
set.seed(125)
library(rpart)
mod1 <- train(Class~., method="rpart", data=training)
print(mod1$finalModel)
plot(mod1$finalModel, uniform=TRUE)
text(mod1$finalModel, use.n=TRUE, all=TRUE, cex=0.5)
# Answer 1: Total=23000 => PS / Total=50000+fiber=10 => WS / Total = 57000 + Fiber=8 ==> PS / Fiber => not possible

# Answer 2: K small: more bias, less variance
# The bias is larger and the variance is smaller. Under leave one out cross validation K is equal to the sample size. 

library(pgmm)
data(olive)
olive = olive[,-1]
head(olive)
mod2 <- train(Area~., method="rpart", data=olive)
print(mod2$finalModel)
plot(mod2$finalModel, uniform=TRUE)
text(mod2$finalModel, use.n=TRUE, all=TRUE, cex=0.5)
predict(mod2, newdata= as.data.frame(t(colMeans(olive))))
# Answer 3: 2.783. It is strange because Area should be a qualitative variable - but tree is reporting the average value of Area as a numeric variable in the leaf predicted for newdata

library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]
set.seed(13234)
mod4 <- glm(chd~age+alcohol+obesity+tobacco+typea+ldl, family="binomial", data=trainSA)
summary(mod4)
missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}
missClass(trainSA$chd, predict(mod4, newdata=trainSA, type="response"))
missClass(testSA$chd, predict(mod4, newdata=testSA, type="response"))
# Answer 4: train=0.27 and test=0.31

library(ElemStatLearn)
data(vowel.train)
data(vowel.test) 
head(vowel.test)
vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)
set.seed(33833)
mod5 <- train(y~., data=vowel.train, method="rf", prox=TRUE)
varImp(mod5)
