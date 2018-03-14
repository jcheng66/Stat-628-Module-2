#load in the libraries we'll need
library(dplyr)
library(tidyverse)
library(tidytext)



#define a negate lexicon

negate_words <- c("not", "without", "no", "cannot","can't", "don", "won","wouldn","couldn","doesn",
                  "never","none","nobody","nothing","neither","n't")

degree_words<-c("almost","absolutely","awfully","badly","barely","completely","decidedly","deeply",
"enough","enormously","entirely","extremely","fairly","far","fully","greatly","hardly","highly","how",
"incredibly","indeed","intensely","just","least","less","little","lots","most","much","nearly",
"perfectly","positively","practically","pretty","purely","quite","rather","really","scarcely",
"simply","so","somewhat","strongly","terribly","thoroughly","too","totally","utterly","very",
"virtually","well")

setwd("/Users/MacBook/Desktop/sentiment")
scorebase<-read.csv("score_base.csv")
train_1<-read.csv("train_data.csv")
nrow(train_1)
scorebase<-as_tibble(scorebase)
review<-read.csv("/Users/MacBook/Desktop/review.csv")

#testing on small dataset "review"
review_1<-as.character(review$text[1])
tokens1<-data_frame(text=as.character(review$text[i]))%>%
  unnest_tokens(word,text) %>%
  inner_join(scorebase)



#testing on sentiment list created by previous lexicons 
mean(sentiment_1[[1]]$score)
median(sentiment_1[[1]]$score)
abs(sum(sentiment_1[[1]]$score[sentiment_1[[1]]$score>0])/sum(sentiment_1[[1]]$score[sentiment_1[[1]]$score<0]))


train_mean<-rep(1,times=nrow(train_1))
sentiment_score<-vector("list",nrow(train_1))
for(i in 1:nrow(train_1)){
  sentiment_score[[i]]<-""
}
# create a list of sentiment words & score based on the scorebase(the lexicon of our own)
for(i in 1:nrow(train_1))
{
  sentiment_score[[i]]<-data_frame(text=as.character(train_1$text[i]))%>%
    unnest_tokens(word,text) %>%
    inner_join(scorebase)
}

# extract the mean score of each review
for(i in 1:nrow(train_1))
{
  train_mean[i]<-mean(sentiment_score[[i]]$score)
}
write.csv(train_mean,"train_mean.csv",row.names=F)

