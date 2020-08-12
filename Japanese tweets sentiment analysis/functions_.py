#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 15:09:39 2020

@author: khan
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
import MeCab
import glob
import re

def scatterplot(x_data, y_data, x_label="", y_label="", title="", color = "r", yscale_log=False):
    _, ax = plt.subplots()
    ax.scatter(x_data, y_data, s = 10, color = color, alpha = 0.75)

    if yscale_log == True:
        ax.set_yscale('log')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
def lineplot(x_data, y_data, x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    ax.plot(x_data, y_data, lw = 2, color = '#539caf', alpha = 1)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def removeSpecialChar(orignal):
    return re.sub("\W+", '', orignal)

def word_tokenization(file, tagger):
    txt = ""
    with open(file, 'r') as f:
        txt += str(removeSpecialChar(f.read()))  
    txt_ls = tagger.parse(txt).split()
    return (" ".join(txt_ls))    

  
def vocab_count_X(df):
    tagger = MeCab.Tagger("-Owakati")
    vectorizer = TfidfVectorizer(min_df=15, max_df=30)

    corpus = []; H=[]; I=[]; J=[]
    for index, d in df.iterrows():
        fn = "tweets/{}.txt".format(d["通し番号"])
        if fn in glob.glob('tweets/*.txt'):
            corpus.append(word_tokenization(fn, tagger))
            H.append(d["一般的信頼合計"]); I.append(d["社会的スキル合計"]); J.append(d["心理的幸福感合計"])
            
    X = vectorizer.fit_transform(corpus)
    return X, H, I, J

# Not a good idea to use
def vocab_count_X_65_users(df):
    tagger = MeCab.Tagger("-Owakati")
    vectorizer = TfidfVectorizer(min_df=15)

    corpus = []; H=[]; I=[]; J=[]
    for index, d in df.iterrows():
        if d['tweet_count'] >= 50:
            fn = "tweets/{}.txt".format(d["通し番号"])
            if fn in glob.glob('tweets/*.txt'):
                corpus.append(word_tokenization(fn, tagger))
                H.append(d["一般的信頼合計"]); I.append(d["社会的スキル合計"]); J.append(d["心理的幸福感合計"])
                
    X = vectorizer.fit_transform(corpus)
    return X, H, I, J

def negative_tweet_porpostion_X(df):
    X = df[df['﻿negative_tweet_propotion'].notna()]['﻿negative_tweet_propotion'].values.tolist()
    X = [[i] for i in X]
    H = df[df['﻿negative_tweet_propotion'].notna()]["一般的信頼合計"].values.tolist()
    I = df[df['﻿negative_tweet_propotion'].notna()]["社会的スキル合計"].values.tolist()
    J = df[df['﻿negative_tweet_propotion'].notna()]["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def negative_tweet_porpostion_X_65_users(df):
    X = df[df['tweet_count'] >= 50]['﻿negative_tweet_propotion'].values.tolist()
    X = [[i] for i in X]
    H = df[df['tweet_count'] >= 50]["一般的信頼合計"].values.tolist()
    I = df[df['tweet_count'] >= 50]["社会的スキル合計"].values.tolist()
    J = df[df['tweet_count'] >= 50]["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def negative_tweet_porpostion_by_neural_net_X_65_users(df):
    X = df[df['>=50 (X)'].notna()]['r_neg_tweets'].values.tolist()
    X = [[i] for i in X]
    H = df[df['>=50 (X)'].notna()]["一般的信頼合計"].values.tolist()
    I = df[df['>=50 (X)'].notna()]["社会的スキル合計"].values.tolist()
    J = df[df['>=50 (X)'].notna()]["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def positive_tweet_porpostion_by_neural_net_X_65_users(df):
    X = df[df['>=50 (X)'].notna()]['r_pos_tweets'].values.tolist()
    X = [[i] for i in X]
    H = df[df['>=50 (X)'].notna()]["一般的信頼合計"].values.tolist()
    I = df[df['>=50 (X)'].notna()]["社会的スキル合計"].values.tolist()
    J = df[df['>=50 (X)'].notna()]["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def positive_negative_tweet_by_neural_net_X_65_users(df):
    X = df.loc[df['>=50 (X)'].notna()][['r_pos_tweets', 'r_neg_tweets']].values.tolist()
    H = df[df['>=50 (X)'].notna()]["一般的信頼合計"].values.tolist()
    I = df[df['>=50 (X)'].notna()]["社会的スキル合計"].values.tolist()
    J = df[df['>=50 (X)'].notna()]["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def tweet_count_X_65_users(df):
    X = df[df['tweet_count'] >= 50]['tweet_count'].values.tolist()
    X = [[i] for i in X]
    H = df[df['tweet_count'] >= 50]["一般的信頼合計"].values.tolist()
    I = df[df['tweet_count'] >= 50]["社会的スキル合計"].values.tolist()
    J = df[df['tweet_count'] >= 50]["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def gender_X(df):
    X = df['性別'].values.tolist()
    X = [[i] for i in X]
    H = df["一般的信頼合計"].values.tolist()
    I = df["社会的スキル合計"].values.tolist()
    J = df["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def gender_X_65_users(df):
    X = df[df['tweet_count'] >= 50]['性別'].values.tolist()
    X = [[i] for i in X]
    H = df[df['tweet_count'] >= 50]["一般的信頼合計"].values.tolist()
    I = df[df['tweet_count'] >= 50]["社会的スキル合計"].values.tolist()
    J = df[df['tweet_count'] >= 50]["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def followers_X(df):
    X = df[df['followers'].notna()]['followers'].values.tolist()
    X = [[i] for i in X]
    H = df[df['followers'].notna()]["一般的信頼合計"].values.tolist()
    I = df[df['followers'].notna()]["社会的スキル合計"].values.tolist()
    J = df[df['followers'].notna()]["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def followers_X_65_users(df):
    X = df[df['tweet_count'] >= 50]['followers'].fillna((df['followers'].mean())).values.tolist()
    X = [[i] for i in X]
    H = df[df['tweet_count'] >= 50]["一般的信頼合計"].values.tolist()
    I = df[df['tweet_count'] >= 50]["社会的スキル合計"].values.tolist()
    J = df[df['tweet_count'] >= 50]["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def following_X(df):
    X = df[df['following'].notna()]['following'].values.tolist()
    X = [[i] for i in X]
    H = df[df['following'].notna()]["一般的信頼合計"].values.tolist()
    I = df[df['following'].notna()]["社会的スキル合計"].values.tolist()
    J = df[df['following'].notna()]["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def following_X_65_users(df):
    X = df[df['tweet_count'] >= 50]['following'].fillna((df['following'].mean())).values.tolist()
    X = [[i] for i in X]
    H = df[df['tweet_count'] >= 50]["一般的信頼合計"].values.tolist()
    I = df[df['tweet_count'] >= 50]["社会的スキル合計"].values.tolist()
    J = df[df['tweet_count'] >= 50]["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def followers_following_X(df):
    X = df.loc[df['following'].notna()][['followers', 'following']].values.tolist()
    H = df[df['following'].notna()]["一般的信頼合計"].values.tolist()
    I = df[df['following'].notna()]["社会的スキル合計"].values.tolist()
    J = df[df['following'].notna()]["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def followers_following_X_65_users(df):
    X1 = df.loc[df['tweet_count'] >= 50]['followers'].fillna((df['followers'].mean())).values.tolist()
    X2 = df.loc[df['tweet_count'] >= 50]['following'].fillna((df['following'].mean())).values.tolist()
    X = [[X1[i], X2[i]] for i in range(len(X1))]
    H = df[df['tweet_count'] >= 50]["一般的信頼合計"].values.tolist()
    I = df[df['tweet_count'] >= 50]["社会的スキル合計"].values.tolist()
    J = df[df['tweet_count'] >= 50]["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def all_numerical_params_X(df):
    df['tweet_count'] = df['tweet_count'].fillna((df['tweet_count'].mean()))
    df['性別'] = df['性別'].fillna((df['性別'].mean()))
    df['followers'] = df['followers'].fillna((df['followers'].mean()))
    df['following'] = df['following'].fillna((df['following'].mean()))
    df['tweet_count'] = df['tweet_count'].fillna((df['tweet_count'].mean()))
    df['﻿negative_tweet_propotion'] = df['﻿negative_tweet_propotion'].fillna((df['﻿negative_tweet_propotion'].mean()))
    X = df[['性別', 'followers', 'following', 'tweet_count', '﻿negative_tweet_propotion']].values.tolist()
    H = df["一般的信頼合計"].values.tolist()
    I = df["社会的スキル合計"].values.tolist()
    J = df["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def all_numerical_params_X_65_users(df1):
    df = df1.loc[df1['tweet_count'] >= 50].copy()
    df['tweet_count'] = df['tweet_count'].fillna((df['tweet_count'].mean()))
    df['性別'] = df['性別'].fillna((df['性別'].mean()))
    df['followers'] = df['followers'].fillna((df['followers'].mean()))
    df['following'] = df['following'].fillna((df['following'].mean()))
    df['tweet_count'] = df['tweet_count'].fillna((df['tweet_count'].mean()))
    df['﻿negative_tweet_propotion'] = df['﻿negative_tweet_propotion'].fillna((df['﻿negative_tweet_propotion'].mean()))
    X = df[['性別', 'followers', 'following', 'tweet_count', '﻿negative_tweet_propotion']].values.tolist()
    H = df["一般的信頼合計"].values.tolist()
    I = df["社会的スキル合計"].values.tolist()
    J = df["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def topic_proportion_X_65_users(df, df_topics):
    X = df_topics.drop(columns=['index','通し番号']).values.tolist()
    H = df[df['tweet_count'] >= 50]["一般的信頼合計"].values.tolist()
    I = df[df['tweet_count'] >= 50]["社会的スキル合計"].values.tolist()
    J = df[df['tweet_count'] >= 50]["心理的幸福感合計"].values.tolist()
    return X, H, I, J

def positive_negative_tweet_topic_proportion_X_65_users(df, df_topics):
    new_df = df_topics.copy()
    new_df['r_pos_tweets'] = df.loc[df['>=50 (X)'].notna()]['r_pos_tweets'].values.tolist()
    new_df['r_neg_tweets'] = df.loc[df['>=50 (X)'].notna()]['r_neg_tweets'].values.tolist()
    X = new_df.drop(columns=['index','通し番号']).values.tolist()
    H = df[df['>=50 (X)'].notna()]["一般的信頼合計"].values.tolist()
    I = df[df['>=50 (X)'].notna()]["社会的スキル合計"].values.tolist()
    J = df[df['>=50 (X)'].notna()]["心理的幸福感合計"].values.tolist()
    return X, H, I, J

