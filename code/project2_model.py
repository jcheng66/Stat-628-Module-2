import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import load_npz
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LassoCV, Lasso
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error

"""text clean"""
text = pd.read_csv("train_data.csv",usecols=[2])
text2 = pd.read_csv("testval_data.csv",usecols=[2])
text3 = text.append(text2,ignore_index=True)
del text, text2
text3 = text3.values.tolist()
text3 = [sent[0] for sent in text3]
stop = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")
def text_clean(sent):
    sent = sent.lower()
    sent = re.sub('[^a-zA-Z]',' ', sent)
    sent = word_tokenize(sent)
    sent = [w for w in sent if len(w) > 1]
    sent = [w for w in sent if not w in stop]
    sent = [stemmer.stem(word) for word in sent]
    sent = " ".join(sent)
    return(sent)
text3 = [text_clean(sent) for sent in text3]

"""save"""
df = pd.DataFrame(text3)
df.to_csv("alltext.csv")
del df

"""load"""
text3 = pd.read_csv("alltext.csv",usecols=[1])
text3 = text3.values.tolist()
text3 = [sent[0] for sent in text3]

"""tfidf"""
vectorizer=CountVectorizer()
transformer=TfidfTransformer()
tfidf=transformer.fit_transform(vectorizer.fit_transform(text3))
scipy.sparse.save_npz('tfidf.npz',tfidf)

"""load y"""
y = pd.read_csv("train_data.csv",usecols=[0])
tfidf = load_npz('tf_20.npz')
x = tfidf[:y.shape[0],]
x_new = tfidf[y.shape[0]:,]
y = np.array(y).reshape(-1)

"""train and test split"""
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=666)

"""single model"""
clf = LogisticRegression(C=1, max_iter=100, tol=1e-4,
                             penalty='l2', solver='sag', verbose=1)
clf.fit(X_train, y_train)
pre = clf.predict(X_test)
mean_squared_error(pre,y_test)

"""ensemble"""
feature = np.zeros((tfidf.shape[0],10))
for i, c in enumerate(np.arange(0.1,2,0.2)):
    clf = LogisticRegression(C=c, max_iter=100, tol=1e-4,
                             multi_class='multinomial',
                             penalty='l2', solver='sag', verbose=1)
    clf.fit(x,y)
    pred1 = clf.predict(x)
    pred2 = clf.predict(x_new)
    pred = np.append(pred1, pred2)
    feature[:,i] = pred
np.save("log_feature.npy")
feature = np.zeros((x.shape[0],5))
for i, c in enumerate(np.arange(0.1,2,0.2)):
    clf = LinearSVR(C=c, verbose=1)
    clf.fit(x,y)
    pred1 = clf.predict(x)
    pred2 = clf.predict(x_new)
    pred = np.append(pred1, pred2)
    feature[:,i] = pred
np.save("svr_feature.npy")
feature1 = np.load("log_feature.npy")
feature2 = np.load("svr_feature.npy")
feature = np.concatenate(feature1,feature2)
feature_mean = np.mean(feature,axis=1)

"""lasso selection"""
lm = LassoCV(cv=5,verbose=1,n_jobs=-1).fit(feature[:1082465,:],y_train)
joblib.dump(lm,"lasso.pkl")
lm = joblib.load("lasso.pkl")
mean_squared_error(lm.predict(feature[1082465:,:]),y_test)**0.5
alpha = lm.alphas_
rmse = []
for i,a in enumerate(alpha):
    lasso = Lasso(alpha = a).fit(feature[:1082465,:],y_train)
    rmse.append(mean_squared_error(lasso.predict(feature[1082465:,:]),y_test)**0.5)
    print(i)
best_alpha = rmse.index(min(rmse))
alpha[47]
lasso = Lasso(alpha = alpha[47]).fit(feature[:1082465,:],y_train)
joblib.dump(lasso,"lasso_best.pkl")
np.where(lasso.coef_>0)
lasso.coef_
lasso.intercept_

"""lasso feature"""
clf = LogisticRegression(C=2.5, max_iter=100, tol=1e-4,
                             multi_class='multinomial',
                             penalty='l2', solver='sag', verbose=1)
clf.fit(x, y)
pred = clf.predict(x_pre)
feature[:,0] = pred
clf = LinearSVR(C=1, epsilon=0.1, verbose=1, max_iter=100)
clf.fit(x, y)
pred = clf.predict(x_pre)
feature[:,1] = pred
one = np.ones(1016664)
X = np.column_stack([one,feature])
"""estimate of lasso 0.06997235 0.59631747 0.37839295"""
beta = np.array([0.06997235, 0.59631747, 0.37839295])
feature_lasso = X.dot(beta)
final = (feature_mean+feature_lasso)/2