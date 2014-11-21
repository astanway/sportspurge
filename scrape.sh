python scrape.py
rm readme.md
echo "![x](https://raw.githubusercontent.com/astanway/sportsfilter/master/filter.jpg)" >> readme.md
echo "" >> readme.md
echo "Filter sports for days. Labeled data continuously scraped from news and \
Twitter sources. Optimized for fewer false positives (classifying non-sports as \
sports) at the expense of missing more sports than necessary, as deleting \
non-sports content is more detrimental than not deleting sports content. \

Current stats:
\`\`\`
> python classify.py \
" >> readme.md
python classify.py >> readme.md
echo "\`\`\`" >> readme.md
echo "![x](https://raw.githubusercontent.com/astanway/sportsfilter/master/roc.png)" >> readme.md
python production_classify.py
curl "localhost/retrain"
#git add .
#git commit -am "Automated data ingest and train"
