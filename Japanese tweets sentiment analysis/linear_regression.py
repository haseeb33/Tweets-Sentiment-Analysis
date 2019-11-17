#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:23:15 2019

@author: khan
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

import pandas as pd
import glob
import re
import MeCab

def removeSpecialChar(orignal):
    return re.sub("\W+", '', orignal)

def word_tokenization(file, tagger):
    txt = ""
    with open(file, 'r') as f:
        txt += str(removeSpecialChar(f.read()))
    txt_ls = tagger.parse(txt).split()
    return (" ".join(txt_ls))

df = pd.read_excel("Final1.xlsx", usecols = ["通し番号","アカウント名", "一般的信頼合計", "社会的スキル合計", "心理的幸福感合計"]).values.tolist()
tagger = MeCab.Tagger("-Owakati")
vectorizer = TfidfVectorizer(min_df=9)

corpus = []; H=[]; I=[]; J=[]
for d in df:
    fn = "tweets/{}.txt".format(d[0])
    if fn in glob.glob('tweets/*.txt'):
        corpus.append(word_tokenization(fn, tagger))
        H.append(d[1]); I.append(d[2]); J.append(d[3])


X = vectorizer.fit_transform(corpus)
#X_train, X_test, H_train, H_test = train_test_split(X, H, test_size=0.1, random_state=0)
reg_H = LinearRegression(fit_intercept=False)
# score for each part
scores_H = cross_val_score(reg_H, X, H, cv=10, scoring="r2")
print(scores_H.mean())

reg_I = LinearRegression(fit_intercept=False)
scores_I = cross_val_score(reg_I, X, I, cv=10, scoring="r2")
print(scores_I.mean())

reg_J = LinearRegression(fit_intercept=False)
scores_J = cross_val_score(reg_J, X, J, cv=10, scoring="r2")
print(scores_J.mean())
