# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 22:28:21 2018

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
os.chdir('D:\\UW-Madison\\17-18 Spring\\Stat 628\\Module 2')

# Read data
filename = 'train_data.csv'
reviews = pd.read_csv(filename)

positive_word = pd.read_csv('positivewords.csv', header = None)
negative_word = pd.read_csv('negativewords.csv', header = None)

# Convert categories to right format
for row in range(reviews.shape[0]):
    reviews.iat[row, 7] = ast.literal_eval(reviews.iloc[row, 7])

# =============================================================================


# =============================================================================
# Define word occurrence counting function
def word_hist_count(dataframe, word, total_num):
    # Create a dict
    word_count = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}

    # Loop through all sentences
    for data in dataframe.itertuples():
        word_count[str(data[1])] += data[3].count(word)

    # Create a dict to store numbers of occurence of each star rating
    for key, value in word_count.items():
        word_count[key] = value/total_num[int(key) - 1]
    
    # Return count list
    return word_count


def score_func(count_dict):
    score_each = 0
    score_point = [-5, -2.5, 0, 2.5, 5]
    for i in range(5):
        score_each += count_dict[str(i + 1)] * score_point[i]
        
    return score_each
# =============================================================================


# =============================================================================
num_text = []
for i in range(5):
    num_text.append(reviews.loc[reviews.stars == (i + 1), :].shape[0])

positive_count = []
for word in positive_word.itertuples():
    positive_count.append(word_hist_count(reviews, word[1], num_text))
    
negative_count = []
for word in negative_word.itertuples():
    negative_count.append(word_hist_count(reviews, word[1], num_text))

for i in range(len(positive_count)):  
    plt.figure(figsize=(24,12))
    plt.bar([1, 2, 3, 4 , 5], list(positive_count[i][1].values()))
    plt.title("Bar plot of " + positive_word.iloc[i, 0] + ' : ' + str(i), fontsize = 28)
    plt.show()


for i in range(200, len(negative_count)):  
    plt.figure(figsize=(24,12))
    plt.bar([1, 2, 3, 4 , 5], list(negative_count[i].values()))
    plt.title("Bar plot of " + negative_word.iloc[i, 0] + ' : ' + str(i), fontsize = 28)
    plt.show()

# =============================================================================


# =============================================================================
positive_pos = [1, 6, 9, 10, 15, 16, 21, 23, 26, 27, 28, 31, 32, 34, 38, 39, 41, 42, 44, 50, 51, 52, 53, 54, 56, 59, 62, 63, 
                64, 65, 66, 69, 70, 71, 76, 78, 79, 83, 91, 92, 95, 96, 97, 102, 104, 108, 109, 110, 112, 
                113, 114, 118, 125, 126, 133, 134, 135, 140, 142, 145, 146, 150, 151, 154, 158, 163, 169, 
                170, 171, 172, 173, 174, 175, 176, 179, 187, 190, 193, 194, 195, 200, 202, 204, 211, 213, 
                217, 219, 220, 223, 224, 225, 228, 229, 233, 234, 238, 241, 245, 251, 256, 258, 259, 263]

positive_neg = [2, 7, 11, 75, 89, 103, 115, 117, 131, 139, 141, 144, 152, 153, 155, 168, 178, 191, 198, 199, 205, 236, 
               237, 243]

positive_all = positive_pos + positive_neg

negative_pos = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 17, 18, 22, 24, 25, 26, 29, 31, 32, 33, 34, 36, 38, 
                39, 40, 41, 42, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 59, 62, 63, 65, 66, 67, 
                68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 84, 85, 86, 88, 89, 90, 91, 92, 94, 95, 
                96, 97, 99, 103, 107, 109, 110, 111, 112, 115, 118, 119, 121, 123, 124, 126, 128, 129, 
                130, 133, 134, 137, 138, 141, 143, 144, 145, 146, 147, 148, 151, 155, 156, 157, 158, 160, 161, 162, 
                163, 164, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 182, 
                183, 184, 185, 186, 187, 189, 190, 191, 192, 194, 195, 196, 198, 199, 200, 201, 202, 205, 
                206, 209, 210, 212, 215, 219, 220, 221, 222, 223, 224]

negative_neg = [14, 61, 87, 98, 108, 149]

negative_all = negative_pos + negative_neg
# =============================================================================


# =============================================================================
score, word_name = [], []
for pos in positive_all:
    word_name.append(positive_word.iloc[pos, 0])
    score.append(score_func(positive_count[pos]))
    
for pos in negative_all:
    word_name.append(negative_word.iloc[pos, 0])
    score.append(score_func(negative_count[pos]))

score_base = pd.DataFrame({'score': score, 'word': word_name})
score_base.to_csv('score_base.csv', index = False)
# =============================================================================
