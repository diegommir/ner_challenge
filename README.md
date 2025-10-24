# ner_challenge
Project to solve Jeopardy Questions classifier challenge.

## Intro
The idea of this project is to train a BERT model capable of selecting questions from a Jeopardy questions dataset.

There will be 3 types of questions to be selected:
- Questions Containing Numbers 
- Questions Containing non-English words 
- Questions Containing "unusual" proper nouns

There are no specific definition of each category, so I've chosen my own.
The criteria I used was the easiest way possible for me to automate this selection.

- For the "numbers" category, the criteria is literal numbers. E.g. 2, 100, 1932, etc.
    - Numbers written on the text, like "two" or "one hundred", will be ignored.
    - This is the easiest category to select and to predict, since it is the best defined one.
- For the "non-english" category, the criteria is to use a database of english words found at
https://github.com/dwyl/english-words.
    - I have chosen to use the "words_alpha" version, which have only word with letters. So no numbers or symbols.
    - This will potentially ignore some common abbreviation, contraptions, acronyms.
- For the "unusual" category, the criteria is to identify the non-English word, and test if it has a capital first letter (camel/title case).

### Limitations
This approach has many limitations on real world applications and servers as a proof of concept only.
In a real world scenario would be preferred to choose the training dataset with the supervision of someone from the business area.
This is even going to make difficult for the model to abstract the idea behind the categories, especially the "non-english" and the "unusual" ones.

## Samples
For the "numbers" category as few as one hundred samples were sufficient to get a good result. As expected, I got 90%+ accuracy and F1 score easily and with few adjustments to parameters.

But for the other two categories I had to use more samples and had to try many different adjustments to get good training results. At the end I've used two hundred to improve the results and reach accuracy levels higher than 80%, 85%.

Probably using even more samples and addressing some of the topics discussed within Limitations, would make the model perform even better 
reaching number above 90%, 95%.

## Hyper Parameters
### Batch Size
The batch size had little influence on the final result.

### Learning Rate
Rates of 0.0005 and bigger showed very bad results, probably due to Catastrophic Forgetting, with score numbers dropping drastically.
Also rates smaller than 0.00005 started to show slightly worse results. So numbers between 0.0002 and 0.00007 showed the best results.

### Epochs
3 to 5 epochs have shown to be sufficient to get good results. Numbers above and lower those ones were not showing any improvements.
