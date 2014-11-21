import requests
import key
import slugify
import json
import os.path
import gzip

sports = {
    "sbnation"   : "https://www.kimonolabs.com/api/9nkui488?apikey=",
    "espn"       : "https://www.kimonolabs.com/api/27ekguv0?apikey=",
    "sportsblog" : "https://www.kimonolabs.com/api/bn5fbkyq?apikey=",
    "reuters"    : "https://www.kimonolabs.com/api/3emx328a?apikey=",
    "cbs"        : "https://www.kimonolabs.com/api/ag84xayu?apikey="
}

nonsports = {
    "reutersbusiness" : "https://www.kimonolabs.com/api/3kr0ti3u?apikey=",
    "reutersworld"    : "https://www.kimonolabs.com/api/9j5x8xgm?apikey=",
    "reuterstech"     : "https://www.kimonolabs.com/api/dfm978fg?apikey=",
    "reuterspolitics" : "https://www.kimonolabs.com/api/bahsfnfg?apikey=",
    "wapoworld"       : "https://www.kimonolabs.com/api/9pb6xvnm?apikey=",
    "fox"             : "https://www.kimonolabs.com/api/5l5wimse?apikey=",
    "techcrunch"      : "https://www.kimonolabs.com/api/8wcwswc4?apikey="
}

sports_twitter = {
    "sportscenter"   : "https://www.kimonolabs.com/api/9d4xtpj8?apikey=",
    "adamschefter"   : "https://www.kimonolabs.com/api/dlvhcuy2?apikey=",
    "espn"           : "https://www.kimonolabs.com/api/5kslzlpa?apikey=",
    "bleacherreport" : "https://www.kimonolabs.com/api/4ofnfdow?apikey=",
    "nflsearch"      : "https://www.kimonolabs.com/api/5d8846oc?apikey=",
    "nbasearch"      : "https://www.kimonolabs.com/api/4ohufdvq?apikey="
    #@BillSimmons
}

nonsports_twitter = {
    "pmarca"        : "https://www.kimonolabs.com/api/2x1uwjjy?apikey=",
    "nitashatiku"   : "https://www.kimonolabs.com/api/9fxfahjc?apikey=",
    "umairh"        : "https://www.kimonolabs.com/api/a0v1j2js?apikey=",
    "yaynickq"      : "https://www.kimonolabs.com/api/4fr6r08i?apikey=",
    "blakehounshell": "https://www.kimonolabs.com/api/7t28fv44?apikey=",
    "brianstelter"  : "https://www.kimonolabs.com/api/3fdd5pis?apikey=",
    "hn"            : "https://www.kimonolabs.com/api/blyl8pqe?apikey="
}

def scrape_twitter(tweetertype):
    if tweetertype == "sports":
    	source = sports_twitter

    if tweetertype == "nonsports":
    	source = nonsports_twitter

    for url in source.itervalues():
        r = json.loads(requests.get(url + key.KEY).content)
        results = r['results']
        for tweet in results['collection1']:
            text = tweet['tweet']['text']
            url = tweet['date']['href']

            if text == "":
                continue

            filename = 'data/' + tweetertype + '/' + url.replace('https://twitter.com/','').replace('/','-') + '.gz'
            if not os.path.isfile(filename):
                print filename
                with gzip.open(filename, 'wb') as f:
                    f.write(text.encode("utf-8"))

def scrape(articletype):
    if articletype == "sports":
    	source = sports

    if articletype == "nonsports":
    	source = nonsports

    for url in source.itervalues():
        try:
            r = json.loads(requests.get(url + key.KEY).content)
            results = r['results']
            for article in results['collection1']:
                try:
                    title = article['title']['text']
                except:
                    title = article['title']

                try:
                    body = article['body']['text']
                except:
                    body = article['body']

                filename = 'data/' + articletype + '/' + slugify.slugify(title) + '.gz'
                if not os.path.isfile(filename):
                    print filename
                    with gzip.open(filename, 'wb') as f:
                        f.write(body.encode("utf-8"))
        except:
            continue

if __name__ == "__main__":
    scrape("sports")
    scrape("nonsports")
    scrape_twitter("sports")
    scrape_twitter("nonsports")
