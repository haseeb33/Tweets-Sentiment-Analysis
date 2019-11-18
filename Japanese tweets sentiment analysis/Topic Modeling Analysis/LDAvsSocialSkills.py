#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:57:24 2019

@author: khan
"""

import pandas as pd
import numpy as np
import re
import sys
import csv
sys.path.insert(1, '/home/khan/topicModels-java')
import pytm
import MeCab
import time
import glob
csv.field_size_limit(sys.maxsize)

start_time = time.time()
tagger = MeCab.Tagger("-Owakati")

def removeUsernames(txt):
    return re.sub(r'@\S+', '', txt)

def removeSpecialChar(txt):
    return re.sub("\W+", '', txt)

def removeURLs(txt):
    return re.sub(r'http\S+', '', txt)

def word_tokenization(txt, tagger):
    txt_ls = tagger.parse(txt).split()
    return (" ".join(txt_ls))


train_ls = [str(i) for i in range(6,28,1)]
corpus = []; count = 0
for d in train_ls:
    fn = "training_tweets/{}.txt".format(d)
    txt = ""
    with open(fn, 'r') as f:
        csv_reader = csv.reader(f, delimiter = '\t')
        for row in csv_reader:
            t = removeSpecialChar(removeURLs(removeUsernames(row[3])))
            corpus.append(word_tokenization(t, tagger))
            count +=1
    print("File {} is done".format(d))

print("Total tweets are", count)
docs = pytm.DocumentSet(corpus, min_df=5, max_df=0.5)
print("Corpus Created")

#Applying LDA on our dataset
n_topics = 100
lda = pytm.SVILDA(n_topics, docs.get_n_vocab())
lda.fit(docs, n_iteration=1000, B=1000, n_inner_iteration=5, n_hyper_iteration=20, J=5)
print("LDA fitted")

topic_list = []
alphas = [lda.get_alpha(k) for k in range(n_topics)]
for k, alpha in enumerate(alphas):
    vocab = docs.get_vocab()
    phi = lda.get_phi(k)
    new_phi = np.around(list(phi), decimals = 3)
    print('topic {0} (alpha = {1})'.format(k, np.around(alpha, decimals = 2)))
    a = sorted(zip(vocab, new_phi), key=lambda x: -x[1])[:50]
    topic_list.append(a)

print("Topics Done")
training_time = time.time() - start_time
start_time1 = time.time()

original_df = pd.read_excel("Final1.xlsx")
df = pd.read_excel("Final1.xlsx", usecols = ["通し番号","アカウント名", "一般的信頼合計", "社会的スキル合計", "心理的幸福感合計"]).values.tolist()
corpus1 = [];
for d in df:
    fn = "tweets/{}.txt".format(d[0])
    if fn in glob.glob('tweets/*.txt'):
        txt = ""
        with open(fn, 'r') as f:
            txt += str(removeSpecialChar(removeURLs(removeUsernames(f.read()))))
        corpus1.append(word_tokenization(txt, tagger))

docs1 = pytm.DocumentSet(corpus1, min_df=5, max_df=0.5)
theta1 = lda.get_theta(docs1)
print("Got theta values")

df = pd.DataFrame(topic_list)
writer = pd.ExcelWriter("NLP_JP100topic.xlsx")
df.to_excel(writer, 'LDA')

df1 = pd.DataFrame(theta1)
df1.to_excel(writer, 'ThetaValues')

df2 = pd.DataFrame([[training_time, "Seconds"]])
df2.to_excel(writer, 'LDATime')
writer.save()

total_testing_time = time.time() - start_time1
print(total_testing_time)
