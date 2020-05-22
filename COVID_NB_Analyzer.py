# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:27:29 2020

@author: timot
"""

from commonFunctions import *

#%% Functions

class COVID_NB_Analyzer:
    
    def __init__(self):
        self.caseData_ = pd.DataFrame()
        self.tweetsDF_ = pd.DataFrame()
        self.totalImpact_ = np.array([])
        self.tweetResults_ = []
        self.countVect_ = CountVectorizer()
        self.tfTransformer_ = TfidfTransformer()
        self.clf_ = MultinomialNB()
    
    def collect_tweets(self, geoLocation, distance, sinceDate, untilDate, querySearch, maxTweets, topTweets):
        self.tweetsDF_ = get_tweets(querySearch, startDate = sinceDate, stopDate = untilDate,\
                          geoLocation = geoLocation, distance = distance,\
                              topTweets = topTweets, numMaxTweets = maxTweets)
            
    def load_tweets(self, directory):
        self.tweetsDF_ = pd.read_csv(directory)
        
    def collect_data(self, url):
        self.caseData_ = pd.read_csv(url)
        rename_date_field(self.caseData_)
        
    def load_data(self, directory):
        self.caseData_ = pd.read_csv(directory)
        rename_date_field(self.caseData_)
    
    def compute_total_impact(self, caseWeight, hospWeight, deathWeight):
        caseCounts = self.caseData_['Cases'].values
        hospCounts = self.caseData_['Hospitalizations'].values
        deathCounts = self.caseData_['Deaths'].values
        self.totalImpact_ = ((caseWeight*caseCounts) + (hospWeight*hospCounts)\
            + (deathWeight*deathCounts)) / (caseWeight + hospWeight + deathWeight)
        return self.totalImpact_

    def correlate_tweets(self, deltaInterval):
        # Incorporate interval
        cumSum = np.cumsum(self.totalImpact_)
        deltaCumsum = [cumSum[i+deltaInterval]-val for i, val in enumerate(cumSum[:-deltaInterval])]
        deltaCounts = np.diff(deltaCumsum)
        caseDates = [str(dt.date()) for dt in pd.to_datetime(self.caseData_['Date'])]
        caseDates = caseDates[:-(deltaInterval+1)]
        #Define change
        changeResult = np.where(deltaCounts > 0, 'increase', 'decrease')
        # Get only available tweets
        tweetDates = [str(dt.date()) for dt in pd.to_datetime(self.tweetsDF_['Date'])]
        validIdx = [idx for idx, val in enumerate(tweetDates) if val in caseDates]
        tweetsDF_short = self.tweetsDF_.iloc[validIdx]
        tweetDates_short = [str(dt.date()) for dt in pd.to_datetime(tweetsDF_short['Date'])]
        # Correlate tweets
        tweetResults = []
        for tweetDate in tweetDates_short:
            result = changeResult[find_idx(caseDates, tweetDate)]
            tweetResults.append(result[0])
        # Print case stats
        numIncreaseDays = len(np.where(deltaCounts > 0)[0])
        numDecreaseDays = len(np.where(deltaCounts < 0)[0])
        dayIncreasePer = numIncreaseDays/(numDecreaseDays + numIncreaseDays)
        print("\nTotal increase days: " + str(numIncreaseDays))
        print("Total decrease days: " + str(numDecreaseDays))
        print("Day increase percentage: %0.2f" % (dayIncreasePer*100))
        # Print tweet stats
        numIncreaseTweets = tweetResults.count('increase')
        numDecreaseTweets = tweetResults.count('decrease')
        tweetIncreasePer = numIncreaseTweets/(numDecreaseTweets + numIncreaseTweets)
        print("\nTotal increase tweets: " + str(numIncreaseTweets))
        print("Total decrease tweets: " + str(numDecreaseTweets))
        print("Tweet increase percentage: %0.2f" % (tweetIncreasePer*100))
        
        self.tweetResults_ = tweetResults
        self.tweetsDF_ = tweetsDF_short
        
        return tweetResults, tweetsDF_short
    
    def create_classifier(self, trainSize, stopwordsList, useIDF, do_downsample, do_stat, numFeatures):
        totalTweets = list(self.tweetsDF_['Text'])
        self.clf_, self.count_vect_, self.tfTransformer_\
        = create_NB_text_classifier(totalTweets, self.tweetResults_, trainSize, stopwordsList,\
                                    useIDF, do_downsample=do_downsample,
                                    do_stat=do_stat, n_features=numFeatures)
    
    def run_prediction(self, geoLocation, distance, txtSearch, numMaxTweets, topTweets, printAll):
        predict_from_tweets(self.clf_, self.count_vect_, self.tfTransformer_,\
            txtSearch, geoLocation, distance, numMaxTweets,\
                topTweets, printAll)
    
    def plot_data(self, deltaInterval):
        uniqueDates, tweetsByDay = count_tweets_by_day(self.tweetsDF_)
        uniqueDates = pd.to_datetime(uniqueDates)
        
        dates = [dt.date() for dt in pd.to_datetime(self.caseData_['Date'])]
        caseCounts = self.caseData_['Cases'].values
        hospCounts = self.caseData_['Hospitalizations'].values
        deathCounts = self.caseData_['Deaths'].values
        
        datesInterval = dates[:-deltaInterval]
        cumSum = np.cumsum(self.totalImpact_)
        deltaCumsum = [cumSum[i+deltaInterval]-val for i, val in enumerate(cumSum[:-deltaInterval])]
        
        diffDates = dates[:-1]
        diffTotalImpact = np.diff(self.totalImpact_)
        diffIntervalDates = datesInterval[:-1]
        diffInterval = np.diff(deltaCumsum)
        
        x_values_plt1 = [uniqueDates, dates, dates, dates, dates, datesInterval]
        y_values_plt1 = [tweetsByDay, caseCounts, hospCounts, deathCounts, self.totalImpact_, deltaCumsum]
        labels_plt1 = ['Tweets', 'Cases', 'Hosp', 'Deaths', 'Total Impact', 'Interval']
        xlabel_plt1 = 'Date'
        ylabel_plt1 = 'Counts'
        title_plt1 = 'New Counts'
        
        x_values_plt2 = [diffDates, diffIntervalDates]
        y_values_plt2 = [diffTotalImpact, diffInterval]
        labels_plt2 = ['Daily', 'Interval']
        xlabel_plt2 = 'Date'
        ylabel_plt2 = 'Change'
        title_plt2 = 'Change By Interval'
        
        plot_values(x_values_plt1, y_values_plt1, labels_plt1, xlabel_plt1, ylabel_plt1, title_plt1, isDates=True)
        plot_values(x_values_plt2, y_values_plt2, labels_plt2, xlabel_plt2, ylabel_plt2, title_plt2, isDates=True)

#%% Run script
    
if __name__ == "__main__":
    
    #%% Inputs:
    
    # Classifier tweet parameters:
    # geoLocation = "40.73, -73.94"
    # distance = "20mi"
    # sinceDate = "2020-02-29"
    # untilDate = "2020-05-11"
    # querySearch = "coronavirus covid"
    # maxTweets = 0
    # topTweets = False
    
    # Tweet and data directories:
    tweetDir = './NYC_Tweets.csv'
    dataDir = './NYC_May11.csv'
    
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
    NB_analyzer.load_tweets(tweetDir)
    NB_analyzer.load_data(dataDir)
    NB_analyzer.compute_total_impact(caseWeight, hospWeight, deathWeight)
    NB_analyzer.correlate_tweets(deltaInterval)
    NB_analyzer.plot_data(deltaInterval)
    NB_analyzer.create_classifier(trainSize, stopwordsList, useIDF, do_downsample, do_stat, numFeatures)
    NB_analyzer.run_prediction(geoLocation_predict, distance_predict, txtSearch_predict,\
                               numMaxTweets_predict, topTweets_predict, printAll)