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
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from operator import itemgetter
import random
import warnings

warnings.filterwarnings("ignore")

#%%

def find_idx(inputList, condition):
    return [idx for idx, val in enumerate(inputList) if val == condition]

def del_idx(inputList, idxs):
    for idx in sorted(idxs, reverse=True):
        del inputList[idx]
    return inputList

def plot_count_and_delta(dates, data, label, doDelta = False, doShow = True, figNum = False):
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
    if doDelta:
        plt.subplot(121)
    plt.plot_date(dates, data, '-', label = label)
    plt.title('New Occurrences')
    plt.xlabel('Date')
    plt.ylabel('Counts')
    plt.xticks(rotation = 70)
    plt.legend()
    if doDelta:
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

def plot_case_data(caseData, doDelta = True, showPlots = True):
    dates = pd.to_datetime(caseData['Date'])
    cases = caseData['Cases'].values
    hosp = caseData['Hospitalizations'].values
    deaths = caseData['Deaths'].values
    figNum = plot_count_and_delta(dates, cases, 'Cases', doDelta = doDelta, doShow = False)
    plot_count_and_delta(dates, hosp, 'Hosps', doDelta = doDelta, doShow = False, figNum = figNum)
    plot_count_and_delta(dates, deaths, 'Deaths', doDelta = doDelta, doShow = showPlots, figNum = figNum)
    return figNum

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
    return totalImpact

def correlate_tweets(tweetsDF, caseData, totalImpact, deltaInterval):
    tweetDates = [str(dt.date()) for dt in pd.to_datetime(tweetsDF['Date'])]
    caseDates = [str(dt.date()) for dt in pd.to_datetime(caseData['Date'])]
    cumSum = np.cumsum(totalImpact)
    deltaCumsum = [cumSum[i+deltaInterval]-val for i, val in enumerate(cumSum[:-deltaInterval])]
    deltaCounts = np.diff(deltaCumsum)
    caseDates = caseDates[:-(deltaInterval+1)]
    numIncreaseDays = len(np.where(deltaCounts > 0)[0])
    numDecreaseDays = len(np.where(deltaCounts < 0)[0])
    dayIncreasePer = numIncreaseDays/(numDecreaseDays + numIncreaseDays)
    changeResult = np.where(deltaCounts > 0, 'increase', 'decrease')
    validIdx = [idx for idx, val in enumerate(tweetDates) if val in caseDates]
    tweetsDF_short = tweetsDF.iloc[validIdx]
    tweetDates_short = [str(dt.date()) for dt in pd.to_datetime(tweetsDF_short['Date'])]
    tweetResults = []
    for tweetDate in tweetDates_short:
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
    return tweetResults, tweetsDF_short

def count_tweets_by_day(tweetsDF):
    tweetDates = [str(dt.date()) for dt in pd.to_datetime(tweetsDF['Date'])]
    uniqueDates = list(set(tweetDates))
    uniqueDates.sort()
    numTweetsByDay = [len(find_idx(tweetDates, date)) for date in uniqueDates]
    return uniqueDates, numTweetsByDay

def transform_text(text, count_vect, tfTransformer):
    count_text = count_vect.transform(text)
    tf_text = tfTransformer.transform(count_text)
    return tf_text

def downsample(xValues, yValues):
    totalList = list(zip(xValues, yValues))
    random.shuffle(totalList)
    xRand, yRand = zip(*totalList)
    xRand = list(xRand)
    yRand = list(yRand)
    uniqueY = list(set(yRand))
    numYs = [yRand.count(y) for y in uniqueY]
    minNum = min(numYs)
    for idx, y in enumerate(uniqueY):
        num_del = numYs[idx]-minNum
        y_idxs = find_idx(yRand, y)
        removeIdxs = y_idxs[:num_del+1]
        yRand = del_idx(yRand, removeIdxs)
        xRand = del_idx(xRand, removeIdxs)
    return xRand, yRand

def create_classifier(totalTweets, totalResults, trainSize, stopwordsList, useIDF, do_downsample = False):
    if do_downsample:
        totalTweets, totalResults = downsample(totalTweets, totalResults)
    tweetTxtTrain, tweetTxtTest, resultsTrain, resultsTest = train_test_split(
        totalTweets, totalResults, train_size = trainSize, random_state = 42)
    count_vect = CountVectorizer(stop_words = stopwordsList)
    train_counts = count_vect.fit_transform(tweetTxtTrain)
    tfTransformer = TfidfTransformer(use_idf = useIDF)
    train_tf = tfTransformer.fit_transform(train_counts)
    clf = MultinomialNB().fit(train_tf, resultsTrain)
    test_tf = transform_text(tweetTxtTest, count_vect, tfTransformer)
    
    predictedResults = clf.predict(test_tf)
    classNames = clf.classes_
    classifier_report = classification_report(resultsTest, predictedResults, labels = classNames)
    print("\nModel results: ")
    print(classifier_report)
    disp = plot_confusion_matrix(clf, test_tf, resultsTest,
                             display_labels = classNames,
                             cmap=plt.cm.Blues)
    title = 'Confusion Matrix'
    disp.ax_.set_title(title)
    plt.show()
    return clf, count_vect, tfTransformer

def show_most_informative(clf, count_vect, n = 10):
    classes = clf.classes_
    features = count_vect.get_feature_names()
    probabilities = np.exp(clf.feature_log_prob_)
    one2two_ratio = np.round(np.divide(probabilities[0], probabilities[1]), 3)
    two2one_ratio = np.round(np.divide(probabilities[1], probabilities[0]), 3)
    top_one2two = (sorted(zip(features, one2two_ratio), key = itemgetter(1)))[:-(n+1):-1]
    top_two2one = (sorted(zip(features, two2one_ratio), key = itemgetter(1)))[:-(n+1):-1]
    label_one2two = classes[0] + ':' + classes[1] 
    label_two2one = classes[1] + ':' + classes[0]
    ratio_dict = {label_one2two: top_one2two, label_two2one: top_two2one}
    print("\nBelow printout gives the most informative words.")
    print("Example -> inc:dec: ('gain', 3.0) indicates 'gain'"\
          + "is 3.0x more likely to appear in an inc tweet vs dec tweet.\n")
    print("{:<35s} {:<35s}".format(label_one2two, label_two2one))
    for one, two in zip(top_one2two, top_two2one):
        print("{:<35s} {:<35s}".format(str(one), str(two)))
    return ratio_dict

def predict(clf, count_vect, tfTransformer,\
            txtSearch, geoLocation, distance, numTestMaxTweets,\
                topTestTweets, printAll):
    predictTweets = get_tweets(txtSearch, geoLocation = geoLocation, \
                               distance = distance, topTweets = topTestTweets,\
                                   numMaxTweets = numTestMaxTweets)
    tweetText = list(predictTweets['Text'])
    tf_text = transform_text(tweetText, count_vect, tfTransformer)
    predictions = clf.predict(tf_text)
    if printAll:
        for idx, prediction in enumerate(predictions):
            print("\nTweet:")
            print(tweetText[idx])
            print("Prediction: " + prediction)
    numInc = list(predictions).count('increase')
    numDec = list(predictions).count('decrease')
    print("\nRatio of predicted tweets (inc/dec): " + str(numInc) + '/' + str(numDec))
    if (numInc/(numInc+numDec)) > 0.75:
        print("Cases predicted to increase...")
    elif (numDec/(numInc+numDec)) > 0.75:
        print("Cases predicted to decrease")
    else:
        print("Too wishy washy... Evaluate more indicators")

#%%
    
if __name__ == "__main__":
    
    #%% Gather tweets
    
    # geoLocation = "40.73, -73.94"
    # distance = "20mi"
    # sinceDate = "2020-02-29"
    # untilDate = "2020-05-11"
    # querySearch = "coronavirus covid"
    # maxTweets = 0
    # topTweets = False
    
    # tweetsDF = get_tweets(querySearch, startDate = sinceDate, stopDate = untilDate,\
    #                       geoLocation = geoLocation, distance = distance,\
    #                           topTweets = topTweets, numMaxTweets = maxTweets)
    # tweetsDF.to_csv('./CSV_Files/NYC_Tweets.csv', index = False)
    
    #%% Load CSV data and update field name
    
    caseData = pd.read_csv('./CSV_Files/NYC_May11.csv')
    caseData = rename_date_field(caseData)
    
    try:
        tweetsDF = pd.read_csv('./CSV_Files/NYC_Tweets.csv')
    except:
        print("Wrong directory or tweets have not been pulled")
        
    #%% Print statistics
    
    figNum = plot_case_data(caseData, doDelta = True, showPlots = True)
    
    #%% Compute weighted count
    
    caseWeight = 1
    hospWeight = 20
    deathWeight = 100
    totalImpact = compute_total_impact(caseData, caseWeight, hospWeight, deathWeight)
    
    figNum = plot_case_data(caseData, doDelta = True, showPlots = False)
    plot_count_and_delta(pd.to_datetime(caseData['Date']),\
                         totalImpact, 'Total Impact', doDelta = True, figNum = figNum)
    
    #%% Correlate tweets
    
    deltaInterval = 3 # days
    tweetResults, tweetsDF_short = correlate_tweets(tweetsDF, caseData, totalImpact, deltaInterval)
    
    #%% Plot tweets by day
    
    uniqueDates, tweetsByDay = count_tweets_by_day(tweetsDF_short, tweetResults)
    
    #%% Plot using the delta invterval
    
    caseDates = [str(dt.date()) for dt in pd.to_datetime(caseData['Date'])]
    cumSum = np.cumsum(totalImpact)
    deltaCumsum = [cumSum[i+deltaInterval]-val for i, val in enumerate(cumSum[:-deltaInterval])]
    caseDatesInterval = caseDates[:-deltaInterval]
    figNum = plot_count_and_delta(pd.to_datetime(caseDates), totalImpact,\
                         'New Count', doDelta = False, doShow = False)
    plot_count_and_delta(pd.to_datetime(caseDates), cumSum,\
                         'Cumulative Counts', doDelta = False, doShow = False, figNum = figNum)
    plot_count_and_delta(pd.to_datetime(caseDatesInterval), deltaCumsum,\
                         'By Interval', doDelta = True, doShow = False,)
    
        
    #%% Create model
    
    trainSize = 0.8
    stopwordsList = stopwords.words('english')
    useIDF = True
    do_downsample = True
    totalTweets = list(tweetsDF_short['Text'])
    clf, count_vect, tfTransformer\
        = create_classifier(totalTweets, tweetResults,\
                            trainSize, stopwordsList, useIDF, do_downsample = do_downsample)
    
    #%% Analyze model
    
    numFeatures = 10
    ratio_dict = show_most_informative(clf, count_vect, n = numFeatures)
    
    #%% Predict from new tweets
    geoLocation = "40.73, -73.94"
    distance = "20mi"
    txtSearch = "coronavirus covid"
    numTestMaxTweets = 10
    topTestTweets = True
    printAll = False
    predict(clf, count_vect, tfTransformer,\
            txtSearch, geoLocation, distance, numTestMaxTweets,\
                topTestTweets, printAll)