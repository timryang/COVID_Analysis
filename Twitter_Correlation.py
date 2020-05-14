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

def rename_date_field(df):
    dfColumns = df.columns
    checkDateField = ['date' in tempStr.lower() for tempStr in dfColumns]
    if checkDateField.count(True) != 1:
        raise ValueError("Check CSV for date fields")
    else:
        df.rename(columns = {dfColumns[checkDateField]: 'Date'}, inplace = True)
    return df

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

def correlate_tweets(tweetsDF, caseData, caseWeight, hospWeight, deathWeight):
    tweetDates = [str(dt.date()) for dt in pd.to_datetime(tweetsDF['Date'])]
    caseDates = [str(dt.date() for dt in pd.to_datetime(caseData['Date']))]
    caseCounts = caseData['Counts'].values
    hospCounts = caseData['Hospitalizations'].values
    deathCounts = caseData['Deaths'].values
    totalImpact = (caseWeight*caseCounts) + (hospWeight*hospCounts)\
        + (deathWeight*deathCounts)
    

#%%
    
if __name__ == "__main__":
    
    #%% Load CSV data and update field name
    
    caseData = pd.read_csv('./CSV_Files/NYC_May11.csv')
    caseData = rename_date_field(caseData)
    
    try:
        tweetsDF = pd.read_csv('./CSV_Files/NYC_Tweets.csv')
    except:
        print("Wrong directory or tweets have not been pulled")
        
    
    #%% Gather tweets
    
    geoLocation = "40.73, -73.94"
    distance = "20mi"
    sinceDate = "2020-02-29"
    untilDate = "2020-05-11"
    querySearch = "coronavirus covid"
    maxTweets = 0
    
    tweetsDF = get_tweets(geoLocation, distance, sinceDate, untilDate, querySearch, maxTweets = 0)
    tweetsDF.to_csv('./CSV_Files/NYC_Tweets.csv', index = False)