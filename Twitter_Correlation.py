# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:32:46 2020

@author: timot
"""

import GetOldTweets3 as got
import pandas as pd
import numpy as np
import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#%%

def get_tweets(geoLocation, distance, sinceDate, untilDate, querySearch, maxTweets = 0):
    tweetCriteria = got.manager.TweetCriteria().setNear(geoLocation)\
        .setWithin(distance).setSince(sinceDate).setUntil(untilDate)\
            .setQuerySearch(querySearch).setMaxTweets(maxTweets)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    tweetsParsed = [[tweet.date, tweet.text, tweet.retweets] for tweet in tweets]
    tweetsDF = pd.DataFrame(tweetsParsed, columns = ['Date', 'Text', 'Retweets'])
    tweetsDF.sort_values(by = ['Date'], inplace = True)
    tweetsDF.reset_index(drop = True, inplace = True)
    return tweetsDF

#%%
    
if __name__ == "__main__":
    
    #%% Load CSV Data
    
    caseData = pd.read_csv('./CSV_Files/NYC_May11.csv')
    
    #%% Gather Tweets
    
    geoLocation = "40.73, -73.94"
    distance = "20mi"
    sinceDate = "2020-02-29"
    untilDate = "2020-05-11"
    querySearch = "coronavirus covid"
    maxTweets = 0
    
    tweetsDF = get_tweets(geoLocation, distance, sinceDate, untilDate, querySearch, maxTweets = 0)
    tweetsDF.to_csv('./CSV_Files/NYC_Tweets.csv', index = False)
    
    