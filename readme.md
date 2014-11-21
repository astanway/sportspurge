![x](https://raw.githubusercontent.com/astanway/sportsfilter/master/filter.jpg)

Filter sports for days. Labeled data continuously scraped from news and Twitter sources. Optimized for fewer false positives (classifying non-sports as sports) at the expense of missing more sports than necessary, as deleting non-sports content is more detrimental than not deleting sports content. 
Current stats:
```
> python classify.py 
Training set size: 192044

MultinomialNB
             precision    recall  f1-score   support

         -1       0.96      0.98      0.97     33749
          1       0.97      0.96      0.97     30265

avg / total       0.97      0.97      0.97     64014

[[32944   805]
 [ 1200 29065]]

LogisticRegression
             precision    recall  f1-score   support

         -1       0.97      0.97      0.97     33749
          1       0.97      0.97      0.97     30265

avg / total       0.97      0.97      0.97     64014

[[32717  1032]
 [  932 29333]]

Ensemble at .9 threshold:
             precision    recall  f1-score   support

         -1       0.92      0.99      0.96     33749
          1       0.99      0.90      0.95     30265

avg / total       0.95      0.95      0.95     64014

[[33510   239]
 [ 2882 27383]]
```
![x](https://raw.githubusercontent.com/astanway/sportsfilter/master/roc.png)
