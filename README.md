# README: API and NLP Project (Reddit)
#### Author: Carlos Wilwayco

## Problem Statement
Using the Pushshift's Application Program Interface (API)[3] I was tasked to collect posts from two different subreddits. The two subreddits I chose to extract posts from were called NewsOfTheStupid and NewsOfTheWeird. I was able to pull 2000 different posts with 1000 being in each of the two subreddits. Upon extraction of posts, I was also tasked to use Natural Language Processing (NLP) for processing text data in order to train multiple models in order to predict which subreddit each post originated from. 

## Executive Summary
My first step in the data science workflow was gathering the data. I was able to get the data with the use of the Pushshift API; which is known for having parameters that are used to get information regarding comments, submissions and subreddits. I was able to retrieve a total of 72 features upon combining these two datasets. My next step was data cleaning, where I utilized two seperate notebooks to clean and prepare both models in parallel (Count Vectorizer/Multinomial Naive Bayes and TFIDF/Support Vector Machine). I first checked and fixed my null values by filling missing data where I encountered my first problem being that the selftext on the posts are missing. I then reindexed columns by turning them into binary labels and changed the data types from objects to strings. This is because the first model I was creating was utilizing Count Vectorizer which requires a vector, so the X variable has to be a pandas Series and not a DataFrame. After that I then chose the 3 most important columns, 'subreddit, 'selftext' and 'title' to help build my model in order to discern what subreddit it came from and what the contents of each individual posts posts are.

After the data cleaning process, my next step was to pre-process the text data in order to utilize them with my models. What I used for the feature extraction process were Count Vectorizer for my first model and Term Frequency, Inverse Document Frequency (TFIDF) for my second. In order to make all words identical, I converted all my characters into lowercase. I then removed non-relevant characters such as ‘!, @, #, $,’ etc. for getting rid of url links on posts. I then tokenized my text by splitting it into a string of separate words and characters. I then removed words that are not relevant such as urls and links that were overlooked as well as any words in the English stopwords list. Oddly spelled or misspelled words had to be taken care of as well, so that they are represented all the same. The main problem I encountered was still the occurance of all the selftext in my data were missing.

My next steps after pre-processing my data was the model building process. The first model I built was a Multinomial Naive Bayes, and my first steps were to set up X (‘title’ column) and y (‘subreddit’ column). To see what I needed to check within the classification problem, I used the value counts method in order to see the ratio of the posts originating from each subreddit . I split the data into training and testing sets, instantiated a Count Vectorizer with the default hyperparameters, fit the vectorizer on our corpus, and transformed the corpus. I then converted the training data into a datarame and plotted the top occurring words (“episode”, “splitsvilla”, and the number “13”) with a horizontal bar chart to visualize occurances. This is where I encountered my second problem which was debugging issues with my pipeline that I planned on using to conduct a Grid Search with Count Vectorizer. I decided to get rid of this idea and simply just vectorize by instantiating Count Vectorizer, fit it to my X_train, then transformed both X Train and Test sets. I instantiated Multinomial Naive Bayes and fit it to my vectorized X_train. I then scored my MultinomialNB model on train, test, and accuracy and plotted a confusion matrix and presented a classification report in order to evaluate my scores further.

The process for building the Support Vector Machine (SVM) model was similar with data cleanup and pre-processing text data up until to the point of building the model itself. Before vectorizing, I utilized the Label Encoder for my y_train and test in order to fit and transform features. I then utilized Tfidf Vectorizer and fit it with the “title” feature. I then transformed my X_train and test sets. I then instantiated an SVM model and fit it with the vectorized X and y training sets. I then was able to score both training and testing sets as well as checking the accuracy score. I also plotted a confusion matrix and persented a classification report in order to evaluate my final scores and ultimately decide which of the two models performed better.

## Table of Contents
1. [Project Files](#./data)
2. [Data Origin](#Data-Origin)
3. [Model Selection](#Model-Selection)
4. [Conclusions and Recommendations](#Conclusions-and-Recommendations)

## Data Origin
Links to the subreddits can be found: 
1. [NewsOfTheStupid](https://www.reddit.com/r/NewsOfTheStupid/)
2. [NewsOfTheWeird](https://www.reddit.com/r/NewsOfTheWeird/)

## Model Selection
MultinomialNB:

Train and Test scores:
|      Train      | 0.7388059701492538 | 
|-----------------|--------------------|
|      Test       | 0.740909090909091  |

|                 | precision | recall | f1-score | support |
|-----------------|-----------|--------|----------|---------|
| NewsOfTheStupid | 0.68      | 0.91   | 0.78     | 330     |
| NewsOfTheWeird  | 0.86      | 0.57   | 0.69     | 330     |
| accuracy        |           |        | 0.74     | 660     |
| macro avg       | 0.77      | 0.74   | 0.73     | 660     |
| weighted avg    | 0.77      | 0.74   | 0.73     | 660     |

SVM:

Train and Test scores:
|      Train      | 0.7432835820895523 | 
|-----------------|--------------------|
|      Test       | 0.7333333333333333 |

|                 | precision | recall | f1-score | support |
|-----------------|-----------|--------|----------|---------|
| NewsOfTheStupid | 0.67      | 0.94   | 0.78     | 330     |
| NewsOfTheWeird  | 0.89      | 0.52   | 0.66     | 330     |
| accuracy        |           |        | 0.73     | 660     |
| macro avg       | 0.78      | 0.73   | 0.72     | 660     |
| weighted avg    | 0.78      | 0.73   | 0.72     | 660     |

## Conclusions and Recommendations

The two models I chose were Multinomial Naive Bayes and Support Vector Machine. MultinomialNB treats its features with the assumption of conditional independence between pairs of a feature.[1] Support Vector Machine trats its features as binary for each class of the data; (data belongs to this class or data does not belong to this class).[2] Out of the two models, I would recommend the SVM model as it performed better even though it was overfitting unlike the MultinomialNB model which was underfit. Also the SVM model is more suitable for this type of classification problem due to the problem being, which subreddit does this post originate from (does this post belong to this subreddit or does it not belong to this subreddit). One of the changes with the model that should be implemented to increase and improve scores are implementing more stop words with pre-processing. Also, due to posts from these subreddits just using solely their titles for predictions and only having url links to articles, I encountered porblems with missing selftext data upon model building. Looking over the data extraction process with the use of the Pushshift API, I plan to search other subreddits that have selftext rather than just a title and url links to a different webpage/article. 

I will be continuing on making these improvements to this model in order to obtain better predictions on post origin as well as overall scores. 

## References
1. [Multionomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
2. [Support Vector Machine](https://scikit-learn.org/stable/modules/svm.html)
3. [Pushshift API](https://github.com/pushshift/api)
