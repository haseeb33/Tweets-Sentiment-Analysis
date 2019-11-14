import pandas as pd
import glob
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


def send_result(annotations):
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude
    count = 0
    negative_tweet = 0

    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        count+=1
        if sentence_sentiment < 0:
            negative_tweet+=1
    print("Total senternces are {}".format(count))
    print('Overall Sentiment: score of {} with negative tweets of {}'.format(
        round(score, 4), round(negative_tweet, 4)))
    print("___________________________________________")
    return(count, score, magnitude, negative_tweet)


def analyze(file):
    """Run a sentiment analysis request on text within a passed filename."""
    client = language.LanguageServiceClient()

    with open(file, 'r') as f:
        # Instantiates a plain text document.
        content = f.read()

    document = types.Document(
        content=content,
        type=enums.Document.Type.PLAIN_TEXT)
    annotations = client.analyze_sentiment(document=document)

    # Print the results
    c,s,m,n = send_result(annotations)
    return(c,s,m,n)

original_df = pd.read_excel("Check.xlsx")
df = pd.read_excel("survey.xlsx", usecols = ["通し番号","アカウント名"]).values.tolist()

total_sentences = []; score = []; magnitude = []; negative_tweets = []

for d in df:
    filename = "tweets/{}.txt".format(d[0])
    if filename in glob.glob('tweets/*.txt'):
        try:
	        c,s,m,n = analyze(fn)
        except:
	        c= "E"; s = "E"; m = "E"; n = "E"

    else:
    	c = ""; s = ""; m = ""; n = ""

    total_sentences.append(c)
    score.append(s)
    magnitude.append(m)
    negative_tweets.append(n)


original_df["total_sentences"] = total_sentences
original_df["score"] = score
original_df["magnitude"] = magnitude
original_df["negative_tweets"] = negative_tweets
original_df.to_excel("Final.xlsx")
