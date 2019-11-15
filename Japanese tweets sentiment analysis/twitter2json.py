from time import sleep
from collections import defaultdict
import os.path
from os.path import expanduser
from datetime import datetime, timedelta
import json
from requests_oauthlib import OAuth1Session
from openpyxl import load_workbook
from wltools.preprocessing.twicab import TwiCab
twicab = TwiCab()
JST = timedelta(hours=9)
from_datetime = datetime(2019, 5, 6, 0, 0, 0)

root_dir = '{0}/credentials/data'.format(expanduser('~'))
workbook_path = '{0}/survey.xlsx'.format(root_dir)
timelines_dir = '{0}/timelines'.format(root_dir)
convs_dir = '{0}/convs'.format(root_dir)

secret_path = '{0}/credentials/twitter/my_cred.json'.format(expanduser('~'))
credentials = json.load(open(secret_path))
twitter = OAuth1Session(credentials['api_key'],
                        credentials['api_secret'],
                        credentials['access_key'],
                        credentials['access_secret'])

wb = load_workbook(filename=workbook_path)
sheet = wb['Sheet']
id_column = [c.column for c in sheet['1'] if c.value == '通し番号'][0]
username_column = [c.column for c in sheet['1'] if c.value == 'アカウント名'][0]
usernames = dict([(r[id_column-1].value, r[username_column-1].value) for r in sheet.rows][1:])

def get_id2user():
    return usernames

def fetch_timeline(username, max_id):
    url = "https://api.twitter.com/1.1/statuses/user_timeline.json"
    params = {'screen_name': username, 'max_id': max_id, 'count': 200, 'exclude_replies': False, 'include_rts': True}
    req = twitter.get(url, params = params)

    if req.status_code == 200:
        return json.loads(req.text)
    else:
        print("ERROR: %d" % req.status_code)
        raise ConnectionError('Twitter API returns error code {0}'.format(req.status_code))

def get_datetime(tweet):
    return datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y') + JST

def is_collected_enough(timeline):
    return len(timeline) > 0 and get_datetime(timeline[-1]) < from_datetime

def collect_timeline(username):
    timeline = fetch_timeline(username, None)
    if len(timeline) == 0:
        return timeline
    while not is_collected_enough(timeline):
        sleep(10)
        fetch = fetch_timeline(username, timeline[-1]['id'])
        if len(fetch) <= 1:
            return timeline
        timeline += fetch[1:]
    return timeline

def get_collected_timelines(since_date=None):
    timelines = {}
    for uid in usernames.keys():
        timeline = []
        filename = '{0}/{1}.txt'.format(timelines_dir, uid)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                timeline = json.load(f)
        timelines[uid] = timeline
        if since_date:
            timelines[uid] = [t for t in timeline if get_datetime(t) >= since_date]
    return timelines

def lookup_tweets(tweet_ids):
    url = "https://api.twitter.com/1.1/statuses/lookup.json"
    params = {'id' : ','.join(str(i) for i in tweet_ids)}
    req = twitter.get(url, params = params)

    if req.status_code == 200:
        return json.loads(req.text)
    else:
        print("ERROR: %d" % req.status_code)
        raise ConnectionError('Twitter API returns error code {0}'.format(req.status_code))

def lookup_tweets_batch(tweet_ids):
    if len(tweet_ids) < 100:
        return lookup_tweets(tweet_ids)
    else:
        num_batches = len(tweet_ids) // 100
        tweets = []
        for batch in range(num_batches):
            tweets.extends(lookup_tweets(tweet_ids[batch*100:(batch+1)*100]))
        tweets.extends(lookup_tweets(tweet_ids[num_batches*100:]))
        return tweets


def is_reply(tweet):
    return twicab.parse(tweet['text'])[0][1] == 'user'

def get_collected_convs(since_date=None):
    uid2convs = {}
    for uid in usernames.keys():
        convs = []
        filename = '{0}/{1}.txt'.format(convs_dir, uid)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                convs = json.load(f)
        uid2convs[uid] = convs
        if since_date:
            uid2convs[uid] = [c for c in convs if get_datetime(c[0]) >= since_date]
    return uid2convs
