# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:50:28 2020

@author: timot
"""
from COVID_NB_Analyzer import COVID_NB_Analyzer
from nltk.corpus import stopwords

#%% Inputs:

loadTweets = True # True if load, false if get new

# Classifier tweet parameters (only used if loadTweets = False):
geoLocation = "40.73, -73.94"
distance = "20mi"
sinceDate = "2020-02-29"
untilDate = "2020-05-11"
querySearch = "coronavirus covid"
maxTweets = 0
topTweets = False

# Tweet and data directories:
tweetDir = './CSV_Files/NYC_Tweets.csv'
dataDir = './CSV_Files/NYC_May11.csv'

# Data weights
caseWeight = 1
hospWeight = 20
deathWeight = 100

# Change interval
deltaInterval = 3 # days

# Classifier parameters
trainSize = 0.8
stopwordsList = stopwords.words('english')
useIDF = True
do_downsample = True
do_stat = True
numFeatures = 10

# New tweet prediction parameters:
geoLocation_predict = "40.73, -73.94"
distance_predict = "20mi"
txtSearch_predict = "coronavirus covid"
numMaxTweets_predict = 10
topTweets_predict = True
printAll = False


#%% Execute

NB_analyzer = COVID_NB_Analyzer()
if loadTweets:
    NB_analyzer.load_tweets(tweetDir)
else:
    NB_analyzer.collect_tweets(geoLocation, distance, sinceDate, untilDate, querySearch, maxTweets, topTweets)
NB_analyzer.load_data(dataDir)
NB_analyzer.compute_total_impact(caseWeight, hospWeight, deathWeight)
NB_analyzer.correlate_tweets(deltaInterval)
NB_analyzer.plot_data(deltaInterval)
NB_analyzer.create_classifier(trainSize, stopwordsList, useIDF, do_downsample, do_stat, numFeatures)
NB_analyzer.run_prediction(geoLocation_predict, distance_predict, txtSearch_predict,\
                           numMaxTweets_predict, topTweets_predict, printAll)