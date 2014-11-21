from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, OneClassSVM
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn import preprocessing
from operator import itemgetter
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from scipy.stats import mode
import numpy as np
import os
import re
import flask
import gzip

sports_count = 0
nonsports_count = 0

def chunks(l, n):
    """
    Utility to chunk documents into tweet-sized arrays by word
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def gather_data():
    """
    Iterate through data files on disk and return a shuffled array of strings,
    ready to be vectorized and split into training and test sets
    """
    global sports_count
    global nonsports_count
    sports = os.listdir('data/sports')
    nonsports = os.listdir('data/nonsports')
    data = []

    for article in sports:
        # If it's twitter, don't limit the words. Otherwise, approximate the
        # number of words in a tweet.
        if "-status-" in article:
            tokenmax = 100
        else:
            tokenmax = 25

        with gzip.open('data/sports/' + article, 'rb') as f:
            f = f.read()

            # No js articles, filter this later
            if "function(" in f:
                pass

            tokenized = f.split(' ')
            tweet_sized = chunks(tokenized, tokenmax)
            for tweet in tweet_sized:
                sports_count += 1
                data.append((' '.join(tweet), 1))

    for article in nonsports:
        if "-status-" in article:
            tokenmax = 100
        else:
            tokenmax = 25
        with gzip.open('data/nonsports/' + article, 'rb') as f:
            f = f.read()

            # No js articles, filter this later
            if "function(" in f:
                pass

            tokenized = f.split(' ')
            if "football" in tokenized:
                continue
            tweet_sized = chunks(tokenized, 26)
            for tweet in tweet_sized:
                nonsports_count += 1
                data.append((' '.join(tweet), -1))

    # Replace all scores (ie, 10-22) with a token
    # If there is a decimal in the first number, it skips (220.2-20
    data = [(re.sub(r' (\d+)-(\d+)', r' SPORTSSCOREREPLACEMENTTOKEN', x[0]), x[1]) for x in data]

    return data

def debug(predicted, test, orig):
    """
    Utility tool to print out misclassified data for algorithm tuning.
    """
    for index, x in enumerate(zip(predicted, test)):
        if x[1] != x[0]:
            if x[1] == -1:
                print "misclassified as sports: " + orig[index]
            else:
                print "misclassfied as news: " + orig[index]

def train_classifiers():
    """
    Called by the webserver on startup to train models and return classifiers
    that can then be called with `predict`
    """
    global sports_count
    global nonsports_count

    labels = (-1, 1)

    data = gather_data()

    data_train = np.array( [x[0] for x in data] )
    label_train = np.array( [x[1] for x in data] )

    vectorizer = CountVectorizer(min_df=1,
                                ngram_range=(1, 2),
                                stop_words='english',
                                strip_accents='unicode')

    data_train = vectorizer.fit_transform(data_train)

    print "Training and saving models..."

    sports_prior = float(sports_count) / (float(sports_count) + float(nonsports_count))
    nonsports_prior = float(nonsports_count) / (float(sports_count) + float(nonsports_count))

    joblib.dump(vectorizer, 'models/vectorizer.joblib')
    joblib.dump(MultinomialNB(fit_prior=True, class_prior=[nonsports_prior, sports_prior]).fit(data_train, label_train), 'models/multinomial.joblib')
    joblib.dump(LinearSVC(class_weight="auto").fit(data_train, label_train), 'models/linearsvc.joblib')
    joblib.dump(LogisticRegression(class_weight="auto").fit(data_train, label_train), 'models/lr.joblib')

    #tweet = np.array( ["ebola obama isis"] )
    #tweet = vectorizer.transform(tweet)
    #print MultinomialNB.predict(tweet)
    #print MultinomialNB.predict_proba(tweet)


if __name__ == "__main__":
    classifiers = train_classifiers()
