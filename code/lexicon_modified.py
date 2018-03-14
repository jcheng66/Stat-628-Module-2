# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 09:49:06 2018

@author: kensh
"""

import time
import os
import re
import nltk
import scipy
import sklearn
import random
import ast
import pandas as pd
import numpy as np
import plotly.plotly as py
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.linalg import lstsq
from sklearn import datasets, linear_model
from scipy.sparse import csr_matrix
from pandas import Series, DataFrame
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import *
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer


# =============================================================================
# Change working directory
os.chdir('/Users/MacBook/Desktop')

# Read data
reviews = pd.read_csv('train_data.csv')
tests = pd.read_csv('testval_data.csv')
scorebase = pd.read_csv('score_base.csv')

# Convert categories to right format
    
for row in range(reviews.shape[0]):
    reviews.iat[row, 7] = ast.literal_eval(reviews.iloc[row, 7])

for row in range(tests.shape[0]):   
    tests.iat[row, 7] = ast.literal_eval(tests.iloc[row, 7])

# =============================================================================
    

# =============================================================================
# Create a dictionary to store each restaurant
restaurant_name = pd.unique(reviews.loc[:, 'name'])
restaurant_record = {}
for name in restaurant_name:
    restaurant_record[name] = []

# Loop through all data
for data in reviews.itertuples():
    restaurant_record[data[2]].append(data[1])
    
# Obtain averge score
restaurant_mean = {}
for key, value in restaurant_record.items():
    restaurant_mean[key] = np.mean(value)


# Create a dictionary to store score for each category
category_score = {}

for data in reviews.itertuples():
    for cat in data[8]:
        if cat not in category_score.keys():
            category_score[cat] = [data[1]]
        else:
            category_score[cat].append(data[1])

# Calculate mean score for each category
category_mean = {}
for key, value in category_score.items():
    category_mean[key] = np.mean(value)

# Create regex tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# Create English stop words list
en_stop = stopwords.words("english")

# =============================================================================


# =============================================================================
# Define a third person words counting function for one review
def num_count_per(sentence, word_list):
    # clean and tokenize document string
    raw = sentence.lower()
    
    # Count third words
    if len(word_list) > 1:
        num_count = 0
        for word in word_list:
            num_count += raw.count(word)
    else:
        num_count =raw.count(word_list)
    
    # Return count
    return num_count
# =============================================================================


# =============================================================================
tests['restaurant_mean_score'] = 0.0
tests['category_mean_score'] = ' '
for data in tests.itertuples():
    cat_list = []
    for item in data[8]:
        cat_list.append(category_mean[item])
    tests.iat[data[0], 9] = cat_list


for i, data in enumerate(tests.itertuples()):
    if data[2] in restaurant_mean.keys():
        tests.iat[data[0], 8] = restaurant_mean[data[2]]
    else:
        tests.iat[data[0], 8] = np.mean(tests.iloc[i, 7])


check_words = [['he', 'she', 'her', 'his', 'him'], '!', '?', ':)', ':(']
col_name = ['third_each', 'exclamation_mark', 'question_mark', 'smiling_face', 'annoying_face']

for i, word_list in enumerate(check_words):
    tests[col_name[i]] = ' '
    for data in tests.itertuples():
        tests.iat[data[0], 10 + i] = num_count_per(check_words[i])
        

feature = reviews.iloc[:, 8:14]
feature.to_csv('test_feature.csv', index = False)

# =============================================================================
