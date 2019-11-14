#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 13:03:24 2018

@author: khan
"""
import json
#from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv
import glob
from datetime import datetime, timedelta
import os
import pandas as pd
import re

JST = timedelta(hours=9)
from_datetime = datetime(2019, 5, 6, 0, 0, 0)

def removeURLs(tweet):
    return re.sub(r'http\S+', '', tweet)

def removeUsernames(tweet):
    return re.sub(r'@\S+', '', tweet)

def get_datetime(tweet):
    return datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y') + JST

def get_collected_timelines(since_date=None):
    timelines = {}
    for uid in usernames.keys():
        timeline = []
        filename = '{0}/{1}.txt'.format("timelines", uid)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                timeline = json.load(f)
        timelines[uid] = timeline
        if since_date:
            timelines[uid] = [t for t in timeline if get_datetime(t) >= since_date]
    return timelines


original_df = pd.read_excel("survey.xlsx")
df = pd.read_excel("survey.xlsx", usecols = ["通し番号","アカウント名"]).values.tolist()
usernames = {}
for d in df:
    usernames[d[0]] = d[1]

id_tweet = get_collected_timelines(from_datetime)
only_tweet = {}
tweet_count = []
for i in id_tweet.keys():
    if len(id_tweet[i])>0:
        row = []
        for j in range(len(id_tweet[i])):
            row.append(id_tweet[i][j]["text"])
        tweet_count.append(len(row))
        # Making a tweets txt file with filename as seriel number
        file = '{0}/{1}.txt'.format("tweets", i)
        with open(file, 'w') as f:
            for item in row:
                item = removeUsernames(removeURLs(item))
                f.write("%s.\n" % item)
    else:
        tweet_count.append(0)


original_df["tweet_count"] = tweet_count
# Creating Check excel file having information of tweet count
original_df.to_excel("Check.xlsx")
print("Job Done")
