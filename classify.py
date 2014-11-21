from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, OneClassSVM
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import preprocessing
from operator import itemgetter
from sklearn.metrics import classification_report, roc_curve, auc
from random import shuffle
from scipy.stats import mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
            tokenized = f.read().split(' ')
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
            tokenized = f.read().split(' ')
            if "football" in tokenized:
                continue
            tweet_sized = chunks(tokenized, 26)
            for tweet in tweet_sized:
                nonsports_count += 1
                data.append((' '.join(tweet), -1))

    # Replace all scores (ie, 10-22) with a token
    # If there is a decimal in the first number, it skips (220.2-20
    data = [(re.sub(r' (\d+)-(\d+)', r' SPORTSSCOREREPLACEMENTTOKEN', x[0]), x[1]) for x in data]
    shuffle(data)

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

def classify(algorithm, **kwargs):
    """
    Run data through any classifier, printing out results as well.
    """
    print "\n" + algorithm.__name__

    # play with pipelines later
    #select = SelectPercentile(score_func=chi2, percentile=5)
    #pipeline = Pipeline([('vect', vectorizer), ('select', select), ('logr', LogisticRegression(tol=1e-8, penalty='l2', C=7))])
    #classifier = pipeline.fit(data_train_orig, label_train)

    classifier = algorithm(**kwargs).fit(data_train, label_train)
    label_predicted = classifier.predict(data_test)

#    print data_train[:, 0]
#    plt.scatter(data_train[:, 0], data_train[:, 1])
#    plt.scatter(data_test[:, 0], data_test[:, 1], alpha=0.6)
#    plt.show()
    #debug(label_predicted, label_test, data_test_orig)

    try:
        fpr, tpr, _ = roc_curve(label_test, classifier.predict_proba(data_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('roc.png')
    except Exception as e:
        pass

    # Normalize if needed
    label_predicted = [int(round(x)) for x in label_predicted]
    label_predicted = [-1 if (x <= -1) else 1 for x in label_predicted]

    proba = classifier.predict_proba(data_test)[:, 1]
    ensemble.append(proba)

    print classification_report(label_test, label_predicted)
    print metrics.confusion_matrix(label_test, label_predicted, labels=labels)

# Prepare the data and vectorize
labels = (-1, 1)
ensemble = []

data = gather_data()

training_size = int(round(len(data) * 0.75))

print 'Training set size: ' + str(training_size)

data_train_orig = np.array( [x[0] for x in data[0:training_size]] )
label_train = np.array( [x[1] for x in data[0:training_size]] )

data_test_orig = np.array( [x[0] for x in data[training_size + 1 : len(data)]] )
label_test = np.array( [x[1] for x in data[training_size + 1 : len(data)]] )

vectorizer = CountVectorizer(min_df=1,
                            ngram_range=(1, 2),
                            stop_words='english',
                            strip_accents='unicode')

data_train = vectorizer.fit_transform(data_train_orig)
data_test = vectorizer.transform(data_test_orig)

#h = .02
#x_min, x_max = data_train.min() - .5, data_train.max() + .5
#y_min, y_max = label_train.min() - .5, label_train.max() + .5
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                         np.arange(y_min, y_max, h))

# Run the classifiers
sports_prior = float(sports_count) / (float(sports_count) + float(nonsports_count))
nonsports_prior = float(nonsports_count) / (float(sports_count) + float(nonsports_count))
classify(MultinomialNB, fit_prior=True, class_prior=[nonsports_prior, sports_prior])
#classify(LinearSVC, class_weight="auto")
classify(LogisticRegression, class_weight="auto")
#classify(OneClassSVM, nu=0.1, kernel="rbf", gamma=0.1)

print "\nEnsemble at .9 threshold:"
final_prediction = []
for prediction in zip(ensemble[0],
                    ensemble[1]):

    avg_proba = float(sum(prediction)) / len(prediction)
    if avg_proba >= .90:
        p = 1
    else:
        p = -1
    final_prediction.append(p)

print classification_report(label_test, final_prediction)
print metrics.confusion_matrix(label_test, final_prediction, labels=labels)
