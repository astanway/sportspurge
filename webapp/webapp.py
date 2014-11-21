import logging
import ujson
import sys
from flask import Flask, request, render_template
from flask.ext.cors import CORS
from sklearn.externals import joblib
import bs4
import numpy as np

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config.from_object(__name__)

WEBAPP_IP = 'localhost'
WEBAPP_PORT = 3000

vectorizer = joblib.load('../models/vectorizer.joblib')

classifiers = [
    joblib.load('../models/multinomial.joblib'),
#    joblib.load('../models/lr.joblib')
]

logger = logging.getLogger("AppLog")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s :: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler = logging.FileHandler('/var/log/sportspurge/sportspurge.log')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())
logger.info('starting on %s:%s' % (WEBAPP_IP, WEBAPP_PORT))

@app.route("/")
def index():
    return render_template("index.html"), 200

@app.route("/retrain")
def retrain():
    try:
        vectorizer = joblib.load('../models/vectorizer.joblib')

        classifiers = [
            joblib.load('../models/multinomial.joblib'),
#            joblib.load('../models/lr.joblib')
        ]

        return "retrained", 200

    except Exception as e:
        logger.error(e)

@app.route("/tweet", methods=['POST'])
def tweet():
    data = request.get_json(force=True)
    keys = data.keys()
    response = []
    try:
        tweets = np.array(data.values())
        tweets = vectorizer.transform(tweets)
        ensemble = []

        avg_proba = []
        for index, classifier in enumerate(classifiers):
            avg_proba.append(classifier.predict_proba(tweets)[:, 1])

        for index, prediction in enumerate(zip(avg_proba[0],
                                              # avg_proba[1]
                                              )):

            avg_proba = float(sum(prediction)) / len(prediction)
            if "sportspurge.com" in request.referrer:
                response.append([avg_proba])
                logger.info("site :: " + str(avg_proba) + " :: " + data[keys[index]].replace('\n',''))
            else:
                if avg_proba >= .85:
                    logger.info("sport :: " + str(avg_proba) + " :: " + data[keys[index]].replace('\n',''))
                    response.append(keys[index])
                else:
                    logger.info("non :: " + str(avg_proba) + " :: " + data[keys[index]].replace('\n',''))

        return ujson.dumps(response), 200

    except Exception as e:
        logger.error(e)
        return "Error", 500

if __name__ == "__main__":
    app.run(WEBAPP_IP, WEBAPP_PORT)
