#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:00:43 2019

@author: khan
"""

import MeCab
import glob
import pandas as pd
from collections import Counter
import re
import operator
import numpy as np

def removeSpecialChar(orignal):
    return re.sub(r'[^\w\s]', ' ', orignal)

def word_tokenization(file, tagger):
    txt = ""
    with open(file, 'r') as f:
        txt += str(removeSpecialChar(f.read()))

    txt_ls = tagger.parse(txt).split()
    txt_frequency = Counter(txt_ls)
    return txt_frequency


original_df = pd.read_excel("Final1.xlsx")
df = pd.read_excel("survey.xlsx", usecols = ["通し番号","アカウント名"]).values.tolist()
tagger = MeCab.Tagger("-Owakati")

All_tweet_word_count = {}

for d in df:
    fn = "tweets/{}.txt".format(d[0])
    if fn in glob.glob('tweets/*.txt'):
        All_tweet_word_count[d[0]] = word_tokenization(fn, tagger)
    else:
    	All_tweet_word_count[d[0]] = "N/A"

Repeated_words = {}
for i in All_tweet_word_count.values():
    if i != 'N/A':
        for j in i:
            if j not in Repeated_words.keys():
                Repeated_words[j] = 1
            else:
                Repeated_words[j] += 1

sorted_word_count = sorted(Repeated_words.items(), key=operator.itemgetter(1), reverse=True)

H_col = original_df[["一般的信頼合計"]].copy()
I_col = original_df[["社会的スキル合計"]].copy()
J_col = original_df[["心理的幸福感合計"]].copy()
new_df = pd.DataFrame(columns= ['used_by', 'word', '一般的信頼合計', '社会的スキル合計', '心理的幸福感合計'])

""" Columns of this new_df are,
used_by: number of users used this word in their tweets
word: word itself
一般的信頼合計: correlation with word frequency with H column
社会的スキル合計: correlation of word frequency with I column
心理的幸福感合計: correlation of word frequency with J column

"""
for i in range(3628): # words used by more than 10 people, total 3628 words
    word = sorted_word_count[i][0]
    ls = [];
    count = 0
    for k, v in All_tweet_word_count.items():
        if v == 'N/A':
            ls.append(np.NaN)
        else:
            if word in v:
                ls.append(v[word])
                count+=1
            else:
                ls.append(np.NaN)
    H_col[word] = ls; I_col[word] = ls; J_col[word] = ls
    a = H_col.corr()[word][0]
    b = I_col.corr()[word][0]
    c = J_col.corr()[word][0]
    new_df.loc[i] = [sorted_word_count[i][1], word, a,b,c]
    del H_col[word]; del I_col[word]; del J_col[word]

new_df = new_df.sort_values(by=['一般的信頼合計','一般的信頼合計','一般的信頼合計'], ascending=False)
new_df.to_excel("Sorterd_Words_Correlation.xlsx")
