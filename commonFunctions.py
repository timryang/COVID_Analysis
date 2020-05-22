# -*- coding: utf-8 -*-
"""
Created on Sat May 16 19:03:33 2020

@author: timot
"""
import GetOldTweets3 as got
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import random
from operator import itemgetter
import warnings
from nltk.corpus import stopwords
import time
import datetime

warnings.filterwarnings("ignore")

#%% Common functions to import

def rename_date_field(df):
    dfColumns = df.columns
    checkDateField = ['date' in tempStr.lower() for tempStr in dfColumns]
    df.rename(columns = {dfColumns[checkDateField][0]: 'Date'}, inplace = True)

def find_idx(input_list, condition):
    return [idx for idx, val in enumerate(input_list) if val == condition]

def del_idx(input_list, idxs):
    for idx in sorted(idxs, reverse=True):
        del input_list[idx]
    return input_list

def get_tweets(txtSearch, startDate=None, stopDate=None, geoLocation=None,\
               distance=None, topTweets=True, numMaxTweets=10):
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

def count_tweets_by_day(tweets_DF):
    tweet_dates = [str(dt.date()) for dt in pd.to_datetime(tweets_DF['Date'])]
    unique_dates = list(set(tweet_dates))
    unique_dates.sort()
    num_daily_tweets = [len(find_idx(tweet_dates, date)) for date in unique_dates]
    return unique_dates, num_daily_tweets

def plot_values(x_values, y_values, labels, x_label, y_label, title, isDates):
    plt.figure(figsize=(20,10))
    if isDates:
        formatter = mdates.DateFormatter("%m-%d")
        locator = mdates.DayLocator(bymonthday=[1, 15])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(locator)
        for idx, i_y_values in enumerate(y_values):
            plt.plot_date(x_values[idx], i_y_values, '-', label=labels[idx])
        plt.xticks(rotation = 70)
    else:
        for idx, i_y_values in enumerate(y_values):
            plt.plot(x_values[idx], i_y_values, '-', label=labels[idx])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def downsample(x_values, y_values):
    total_list = list(zip(x_values, y_values))
    random.shuffle(total_list)
    x_rand, y_rand = zip(*total_list)
    x_rand = list(x_rand)
    y_rand = list(y_rand)
    y_unique = list(set(y_rand))
    num_ys = [y_rand.count(y) for y in y_unique]
    min_num = min(num_ys)
    for idx, y in enumerate(y_unique):
        num_del = num_ys[idx]-min_num
        y_idxs = find_idx(y_rand, y)
        remove_idxs = y_idxs[:num_del+1]
        y_rand = del_idx(y_rand, remove_idxs)
        x_rand = del_idx(x_rand, remove_idxs)
    return x_rand, y_rand

def transform_text(text, count_vect, tfTransformer):
    count_text = count_vect.transform(text)
    tf_text = tfTransformer.transform(count_text)
    return tf_text

def create_NB_text_classifier(x_total, y_total, trainSize, stopwordsList, useIDF,\
                              do_downsample=False, do_stat=True, n_features=10):
    if do_downsample:
        x_total, y_total = downsample(x_total, y_total)
    x_train, x_test, y_train, y_test = train_test_split(
        x_total, y_total, train_size=trainSize, random_state=42)
    count_vect = CountVectorizer(stop_words = stopwordsList)
    x_train_counts = count_vect.fit_transform(x_train)
    tfTransformer = TfidfTransformer(use_idf=useIDF)
    x_train_tf = tfTransformer.fit_transform(x_train_counts)
    clf = MultinomialNB().fit(x_train_tf, y_train)
    
    if do_stat:
        x_test_tf = transform_text(x_test, count_vect, tfTransformer)
        classifier_statistics(x_test_tf, y_test, clf, count_vect, n_features=n_features)
    return clf, count_vect, tfTransformer

def classifier_statistics(x_test, y_truth, clf, count_vect, n_features=10):
    y_predict = clf.predict(x_test)
    class_names = clf.classes_
    report = classification_report(y_truth, y_predict, labels=class_names)
    print("\nModel Results: ")
    print(report)
    disp = plot_confusion_matrix(clf, x_test, y_truth,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues)
    disp.ax_.set_title('Confusion Matrix')
    plt.show()
    NB_show_most_informative(clf, count_vect, n_features=n_features)

def NB_show_most_informative(clf, count_vect, n_features=10):
    classes = clf.classes_
    num_classes = len(classes)
    features = count_vect.get_feature_names()
    probs = np.exp(clf.feature_log_prob_)
    top_features_list = []
    for idx in range(num_classes):
        complement_idx = np.where(np.array([range(num_classes)]) != idx)[1]
        complement_sum_probs = np.sum(probs[complement_idx], axis = 0)
        ratio = np.round(np.divide(probs[idx], complement_sum_probs), 3)
        top_features = (sorted(zip(features, ratio), key = itemgetter(1)))[:-(n_features+1):-1]
        top_features_list.append(top_features)
    print("\nBelow printout gives the most informative words.")
    print("Example -> pos: ('gain', 3) indicates 'gain' is 3x more likely"\
          + " to predict pos compared to the sum prob of other classes.\n")
    for i_class in classes:
        print("{:<35s}".format(i_class), end = '')
    print("\n")
    for idx_f in range(n_features):
        for idx_c in range(num_classes):
            print("{:<35s}".format(str(top_features_list[idx_c][idx_f])), end = '')
        print("")
              
def predict_from_tweets(clf, count_vect, tfTransformer, txt_search,\
                        geo_location=None, distance=None, num_max_tweets=0,\
                            top_tweets=True, printAll=False):
    predictTweets = get_tweets(txt_search, geoLocation=geo_location, \
                                distance=distance, topTweets=top_tweets,\
                                    numMaxTweets=num_max_tweets)
    tweetText = list(predictTweets['Text'])
    tf_text = transform_text(tweetText, count_vect, tfTransformer)
    predictions = clf.predict(tf_text)
    if printAll:
        for idx, prediction in enumerate(predictions):
            print("\nTweet:")
            print(tweetText[idx])
            print("Prediction: " + prediction)
    classes = clf.classes_
    class_counts = []
    for i_class in classes:
        i_num = list(predictions).count(i_class)
        class_counts.append(i_num)
    print("\nPredicted tweets:")
    for idx, i_class in enumerate(classes):
        print(i_class + ": " + str(class_counts[idx]))