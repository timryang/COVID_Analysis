# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:32:46 2020

@author: timot
"""

import GetOldTweets3 as got
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CouplntVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#%%

def find_idx(inputList, condition):
    return [idx for idx, val in enumerate(inputList) if val == condition]

def plot_count_and_delta(dates, data, label, doShow = True, figNum = False):
    dataDelta = np.diff(data)
    formatter = mdates.DateFormatter("%m-%d")
    locator = mdates.DayLocator(bymonthday = [1, 15])
    if figNum:
        plt.figure(figNum)
    else:
        plt.figure(figsize = (20,10))
        ax = plt.gca()
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(locator)
    plt.subplot(121)
    plt.plot_date(dates, data, '-', label = label)
    plt.title('New Occurrences')
    plt.xlabel('Date')
    plt.ylabel('Counts')
    plt.xticks(rotation = 70)
    plt.legend()
    plt.subplot(122)
    plt.plot_date(dates[:-1], dataDelta, '-', label = label)
    plt.title('Delta Occurrences')
    plt.xlabel('Date')
    plt.ylabel('Counts')
    plt.xticks(rotation = 70)
    plt.legend()
    if doShow:
        plt.show()
    return plt.gcf().number

def plot_case_data(caseData, totalImpact = []):
    dates = pd.to_datetime(caseData['Date'])
    cases = caseData['Cases'].values
    hosp = caseData['Hospitalizations'].values
    deaths = caseData['Deaths'].values
    figNum = plot_count_and_delta(dates, cases, 'Cases', doShow = False)
    plot_count_and_delta(dates, hosp, 'Hosps', doShow = False, figNum = figNum)
    if not list(totalImpact):
        plot_count_and_delta(dates, deaths, 'Deaths', figNum = figNum)
    else:
        plot_count_and_delta(dates, deaths, 'Deaths', doShow = False, figNum = figNum)
        plot_count_and_delta(dates, totalImpact, 'Total Impact', figNum = figNum)

def rename_date_field(df):
    dfColumns = df.columns
    checkDateField = ['date' in tempStr.lower() for tempStr in dfColumns]
    if checkDateField.count(True) != 1:
        raise ValueError("Check CSV for date fields")
    else:
        df.rename(columns = {dfColumns[checkDateField][0]: 'Date'}, inplace = True)
    return df
    
def get_tweets(txtSearch, startDate = None, stopDate = None, geoLocation = None,\
               distance = None, topTweets = True, numMaxTweets = 10):
    if (startDate == None and geoLocation == None):
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(txtSearch)\
                                                .setTopTweets(topTweets)\
                                                .setMaxTweets(numMaxTweets)
    elif (geoLocation == None and startDate != None):
        tweetCriteria = got.manager.TweetCriteria().setSince(startDate)\
                                                .setUntil(stopDate)\
                                                .setQuerySearch(txtSearch)\
                                                .setTopTweets(topTweets)\
                                                .setMaxTweets(numMaxTweets)
    elif (startDate == None and geoLocation != None):
        tweetCriteria = got.manager.TweetCriteria().setNear(geoLocation)\
                                                .setWithin(distance)\
                                                .setQuerySearch(txtSearch)\
                                                .setTopTweets(topTweets)\
                                                .setMaxTweets(numMaxTweets)
    else:
        tweetCriteria = got.manager.TweetCriteria().setSince(startDate)\
                                                .setUntil(stopDate)\
                                                .setNear(geoLocation)\
                                                .setWithin(distance)\
                                                .setQuerySearch(txtSearch)\
                                                .setTopTweets(topTweets)\
                                                .setMaxTweets(numMaxTweets)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    tweetsParsed = [[tweet.date, tweet.text, tweet.retweets] for tweet in tweets]
    tweetsDF = pd.DataFrame(tweetsParsed, columns = ['Date', 'Text', 'Retweets'])
    tweetsDF.sort_values(by = ['Date'], inplace = True)
    tweetsDF.reset_index(drop = True, inplace = True)
    return tweetsDF

def compute_total_impact(caseData, caseWeight, hospWeight, deathWeight):
    caseCounts = caseData['Cases'].values
    hospCounts = caseData['Hospitalizations'].values
    deathCounts = caseData['Deaths'].values
    totalImpact = ((caseWeight*caseCounts) + (hospWeight*hospCounts)\
        + (deathWeight*deathCounts)) / (caseWeight + hospWeight + deathWeight)
    # totalImpact = (caseWeight*caseCounts) + (hospWeight*hospCounts)\
    #     + (deathWeight*deathCounts)
    plot_case_data(caseData, totalImpact)
    return totalImpact

def correlate_tweets(tweetsDF, caseData, totalImpact):
    tweetDates = [str(dt.date()) for dt in pd.to_datetime(tweetsDF['Date'])]
    caseDates = [str(dt.date()) for dt in pd.to_datetime(caseData['Date'])]
    deltaCounts = np.diff(totalImpact)
    numIncreaseDays = len(np.where(deltaCounts > 0)[0])
    numDecreaseDays = len(np.where(deltaCounts < 0)[0])
    dayIncreasePer = numIncreaseDays/(numDecreaseDays + numIncreaseDays)
    changeResult = np.where(deltaCounts > 0, 'increase', 'decrease')
    tweetResults = []
    for tweetDate in tweetDates:
        result = changeResult[find_idx(caseDates, tweetDate)]
        tweetResults.append(result[0])
    numIncreaseTweets = tweetResults.count('increase')
    numDecreaseTweets = tweetResults.count('decrease')
    tweetIncreasePer = numIncreaseTweets/(numDecreaseTweets + numIncreaseTweets)
    print("Total increase days: " + str(numIncreaseDays))
    print("Total decrease days: " + str(numDecreaseDays))
    print("Day increase percentage: %0.2f" % (dayIncreasePer*100))
    print("\nTotal increase tweets: " + str(numIncreaseTweets))
    print("Total decrease tweets: " + str(numDecreaseTweets))
    print("Tweet increase percentage: %0.2f" % (tweetIncreasePer*100))
    return tweetResults

#%%
    
if __name__ == "__main__":
    
    #%% Load CSV data and update field name
    
    caseData = pd.read_csv('./CSV_Files/NYC_May11.csv')
    caseData = rename_date_field(caseData)
    
    try:
        tweetsDF = pd.read_csv('./CSV_Files/NYC_Tweets.csv')
    except:
        print("Wrong directory or tweets have not been pulled")
        
    #%% Print statistics
    
    plot_case_data(caseData)
    
    #%% Compute weighted count
    
    caseWeight = 1
    hospWeight = 20
    deathWeight = 100
    totalImpact = compute_total_impact(caseData, caseWeight, hospWeight, deathWeight)
    
    #%% Correlate tweets
    
    tweetResults = correlate_tweets(tweetsDF, caseData, totalImpact)
            
    #%% Gather tweets
    
    geoLocation = "40.73, -73.94"
    distance = "20mi"
    sinceDate = "2020-02-29"
    untilDate = "2020-05-11"
    querySearch = "coronavirus covid"
    maxTweets = 0
    topTweets = False
    
    tweetsDF = get_tweets(querySearch, startDate = sinceDate, stopDate = untilDate,\
                          geoLocation = geoLocation, distance = distance,\
                              topTweets = topTweets, numMaxTweets = maxTweets)
    tweetsDF.to_csv('./CSV_Files/NYC_Tweets.csv', index = False)