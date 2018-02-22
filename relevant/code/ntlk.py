import numpy as np
import pandas as pd
import lightgbm as lgb
import gensim
import keras
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from random import shuffle
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import Conv1D,GlobalMaxPooling1D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import rmsprop
from keras.layers.normalization import BatchNormalization

df = pd.read_csv("D:/study/DS2-2/Data Science/model2/train_data_sub.csv")# first 10000 sample from training is sub
text = df.iloc[:,3]#sub 3 full 2
star = df.iloc[:,1]#sub 1 full 0
tokenizer = RegexpTokenizer(r'\w+')
ps = PorterStemmer()
text1 = []
for sent in text:
    sent = sent.lower()
    sent = tokenizer.tokenize(sent)

    stop = set(stopwords.words("english"))
    xstop = set(
        ['aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'no', 'not', 'shouldn', 'wasn',
         'weren', 'wouldn'])
    word21 = []
    for w in sent:
        if not w in stop - xstop:
            if w in xstop:
                word21.append('not')
            else:
                w = ps.stem(w)
                word21.append(w)
    sent = word21
    text1.append(sent)
del df   #end cnn

text2 = []
for i, sent in enumerate(text1):
    text2.append([sent , ['label'+str(i)]])
text3 = []
for sent, label in text2:
    text3.append(TaggedDocument(sent,label))
del text2

def sentences_perm(sentences):
    shuffle(sentences)
    return sentences

model = Doc2Vec(min_count=10, window=5, vector_size=100, sample=1e-4, negative=5, workers=8)
model.build_vocab(text3)
token_count = sum([len(sentence) for sentence in text3])
model.train(text3, total_examples=token_count, epochs=10)


model.save('yelp.d2v')
train_arrays = np.zeros((10000, 100))
train_labels = np.zeros(10000)
train_labels = np.array(star)
for i in range(10000):
    train_arrays[i] = model['label'+str(i)]
X_train, X_test, y_train, y_test = train_test_split(train_arrays,train_labels, test_size=0.2, random_state=666)
clf = tree.DecisionTreeRegressor(max_depth=5).fit(X_train,y_train)
pre = clf.predict(X_test)
mean_squared_error(y_test, pre.round())#1.38
sum(y_test==pre.round())/len(pre)

#lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric':'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                valid_sets=lgb_eval,  # eval training data
                )

#rf
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train, y_train)
pre = clf.predict(X_test)
mean_squared_error(y_test, pre)#2.14

#svm
clf = svm.SVC()
clf.fit(X_train, y_train)
pre = clf.predict(X_test)
mean_squared_error(y_test, pre)


#tfidf
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
dct = Dictionary(text1)
corpus = [dct.doc2bow(line) for line in text1]
model = TfidfModel(corpus)
model.save("tfidf")
corpus_tfidf = model[corpus]

#CNN
textlen = pd.DataFrame([len(x) for x in text1])
textlen.describe()
review_len = 100
review_dim = 100
w2v = gensim.models.Word2Vec(text1, min_count=1,workers=8)
x = np.zeros((10000,100,100))
for i,sent in enumerate(text1):
    for j,word in enumerate(sent):
        if j == 100:
            break
        x[i,j] = w2v[word]
y = np.array(df.iloc[:,1])
np.save('x.npy',x)
np.save('y.npy',y)
y = np_utils.to_categorical(y-1, num_classes=5)
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=666)
model = Sequential()
model.add(Conv1D(input_shape=(100,100),
                 filters = 40,
                 kernel_size = 3,
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(PReLU())
model.add(Dense(5))
model.add(Activation('sigmoid'))
opt = rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
hist = model.fit(X_train,
                y_train,
                batch_size=32,
                epochs = 1000,
                validation_split=0.7,
                shuffle=True,
                callbacks=[earlyStopping])
score = model.evaluate(X_test, y_test, verbose=1)
pre = model.predict(X_test)
mean_squared_error(np.argmax(y_test, axis=1), np.argmax(pre, axis=1))#1.57

#2nd cnn
model = Sequential()
model.add(Conv1D(input_shape=(100,100),
                 filters = 100,
                 kernel_size = 3,
                 activation='relu',
                 strides=1))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv1D(filters = 40,
                 kernel_size = 3,
                 activation='relu',
                 strides=1))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(GlobalMaxPooling1D())
model.add(Dense(5))
model.add(Activation('sigmoid'))
opt = rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
hist = model.fit(X_train,
                y_train,
                batch_size=32,
                epochs = 1000,
                validation_split=0.7,
                shuffle=True,
                callbacks=[earlyStopping])
print(hist.history)
pre = model.predict(X_test)
mean_squared_error(np.argmax(y_test, axis=1), np.argmax(pre, axis=1))#1.51






