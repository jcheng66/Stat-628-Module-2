library(data.table)
library(tm)
library(SnowballC)
library(Matrix)
library(SIS)
library(glmnet)
library(randomForest)
library(xgboost)
setwd("D:/study/DS2-2/Data Science/model2")
df = fread("train_data_sub.csv")
dim(df)
colnames(df)
docs = df$text
str(docs)
docs <- VCorpus(VectorSource(docs))
docs_map <- tm_map(docs,stripWhitespace)
docs_map <- tm_map(docs_map,removePunctuation)
tolower_wrapped <- content_transformer(tolower)
docs_map <- tm_map(docs_map,tolower_wrapped)
docs_map <- tm_map(docs_map,removeNumbers)
docs_map <- tm_map(docs_map,removeWords,stopwords(kind="en"))
docs_map <- tm_map(docs_map,stripWhitespace)
docs_map <- tm_map(docs_map,stemDocument,language="en")
dtm <- DocumentTermMatrix(docs_map)
dtm <- DocumentTermMatrix(docs_map,control = list(weighting=weightTfIdf))#might not be good
mat = Matrix(0, nrow=dtm$nrow, ncol=dtm$ncol)
mat[cbind(dtm$i, dtm$j)] <- dtm$v
#matf = as.matrix(mat)
#matf = scale(matf)
#mat = mat %*% Matrix::Diagonal(x = sqrt(dtm$nrow) / sqrt(Matrix::colSums(mat^2)))
star = df$stars
dtm$ncol
w = numeric(dtm$ncol)
for(i in 1:dtm$ncol){
  w[i] = abs(mat[,i]%*%star)
}
kw = order(w,decreasing = T)[1:200]
kmat = mat[,kw]
kmatf = as.matrix(kmat)
#glmnet
m = cv.glmnet(x=kmatf,y=star,nfolds = 5)
m1 = glmnet(x=kmatf,y=star,lambda = m$lambda.min)
sqrt(sum((round(predict(m1,kmatf))-star)^2)/10000)#1.075
m2 = glmnet(x=x_train,y=y_train,lambda = m$lambda.1se)
sqrt(sum((round(predict(m1,x_test))-y_test)^2)/3000)#1.06
testid = sample(10000,3000)
x_train = kmatf[-testid,]
x_train1 = mat[-testid,]
y_train = star[-testid]
x_test = kmatf[testid,]
x_test1 = mat[-testid,]
y_test = star[testid]
#rf
rf = randomForest(x=x_train,y=factor(y_train),ntree = 500)
sqrt(sum((as.numeric(predict(rf,x_test))-y_test)^2)/3000)#1.28
#xgboost
dtrain <- xgb.DMatrix(x_train,label = y_train)
dtest <- xgb.DMatrix(x_test,label = y_test) 
dtrain1 <- xgb.DMatrix(x_train1,label = y_train)
dtest1 <- xgb.DMatrix(x_test1,label = y_test) 
param <- list(gamma=0, max_depth = 5,
              min_child_weight = 1, silent = 1,eta = 0.2)
set.seed(666)
cv <- xgb.cv(params=param,data = dtrain, nrounds = 500, nfold = 5,
             eval_metric="error", objective ="reg:linear",
             early_stopping_rounds=10,missing=NA)
watchlist <- list(train=dtrain, test=dtest)
bst <- xgb.train(data=dtrain, max_depth=2, eta=1, nrounds=2, watchlist=watchlist,
                 nthread = 2, objective = "reg:linear")
x = xgboost(x_train,y_train,watchlist, nrounds = 200,early_stopping_round = 3)
