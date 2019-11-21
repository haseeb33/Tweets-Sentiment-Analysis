#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:27:17 2019

@author: khan
"""
import pandas as pd
import numpy as np
import re
import sys
sys.path.insert(1, '/home/khan/topicModels-java')
import pytm
import MeCab
import time
import glob
import csv
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

#global dictonary for hashtags and tweets
All_hashtags = {}
All_hashtag_count = {}

def removeHashtags(tweet):
    ls_tweet = tweet.split()
    hashtags = []
    for word in ls_tweet:
        if word[0] == "#":
            hashtags.append(word[1:])
    if hashtags:
        for tag in hashtags:            
            ls_tweet.remove("#" + tag)
    return hashtags, removeSpecialChar(" ".join(ls_tweet))
    
def detectHashtag(tweet):
    hashtags, new_tweet = removeHashtags(tweet)
    if hashtags:
        for tag in hashtags:
            if tag in All_hashtags.keys():
                All_hashtags[tag] = All_hashtags[tag] + " " + word_tokenization(new_tweet, tagger)
                All_hashtag_count[tag] += 1
            else:
                All_hashtags[tag] = word_tokenization(new_tweet, tagger)
                All_hashtag_count[tag] = 1
              
train_ls = [str(i) for i in range(6,28,1)]
for d in train_ls:
    fn = "training_tweets/{}".format(d)
    txt = ""
    with open(fn, 'r') as f:
        csv_reader = csv.reader(f, delimiter = '\t')
        for row in csv_reader:
            t = removeURLs(removeUsernames(row[3]))
            detectHashtag(t)
            
    print("File {} is done".format(d))

print("Total documents are", sum(All_hashtag_count.values()))

corpus = []; count = 0
for key in All_hashtags.keys():
    corpus.append(All_hashtags[key])
    
print("Total documents are", count)
      
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
    a = sorted(zip(vocab, new_phi), key=lambda x: -x[1])[:50]
    topic_list.append(a)

print("Topics Done")
training_time = time.time() - start_time

original_df = pd.read_excel("Final1.xlsx")
df = pd.read_excel("Final1.xlsx", usecols = ["通し番号", "一般的信頼合計", "社会的スキル合計", "心理的幸福感合計"]).values.tolist()

extract_top_50 = pd.read_excel("result_sentiment_analysis.xlsx")
a = extract_top_50.loc[extract_top_50[">=50"]>=0, "通し番号"]
more_than_50_tweets_users = [i for i in a]

imp_df = original_df.loc[original_df["通し番号"].isin(a.to_frame()["通し番号"])]
H_col = imp_df["一般的信頼合計"].values.tolist()
I_col = imp_df["社会的スキル合計"].values.tolist()
J_col = imp_df["心理的幸福感合計"].values.tolist()

corpus1 = []; H=[]; I=[]; J=[]
for d in df:
    if d[0] in more_than_50_tweets_users:
        fn = "tweets/{}.txt".format(d[0])
        if fn in glob.glob('tweets/*.txt'):
            txt = ""
            with open(fn, 'r') as f:
                txt += str(removeSpecialChar(removeURLs(removeUsernames(f.read()))))
            corpus1.append(word_tokenization(txt, tagger))
            H.append(d[1]); I.append(d[2]); J.append(d[3])

docs1 = pytm.DocumentSet(corpus1, min_df=5, max_df=0.5)    
theta1 = lda.get_theta(docs1)
print("Got theta values")

df1 = pd.DataFrame(theta1)
df2 = pd.DataFrame([[training_time, "Seconds"]])

theta_df = df1
H_col = imp_df["一般的信頼合計"].values.tolist()
I_col = imp_df["社会的スキル合計"].values.tolist()
J_col = imp_df["心理的幸福感合計"].values.tolist()

theta_df["H"] = H_col
theta_df["I"] = I_col
theta_df["J"] = J_col
corr = theta_df.corr()

df3 = pd.DataFrame([corr["H"], corr["I"], corr["J"]])
df3 = df3.T
df3 = df3.sort_values(by=['H','I','J'], ascending=False)    

df = pd.DataFrame(topic_list)
writer = pd.ExcelWriter("Hashtag_NLP_JP100Topics65users.xlsx")
df.to_excel(writer, 'LDA')
df1.to_excel(writer, 'ThetaValues')
df2.to_excel(writer, 'LDATime')
df3.to_excel(writer, "CorrWithTopics")
writer.save()
print("Job Done")




