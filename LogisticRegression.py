# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 03:12:26 2019

@author: SRIJAN
"""

import numpy as np
import pandas as pd

df = pd.read_csv('dataset/train_E6oV3lV.csv', encoding='latin-1')
df.head()

df['length'] = df['tweet'].apply(len)

hate_words = ['hate', 'kill', 'racist', 'racism', 'black', 'motherfucker', 'white','libtard', 'amp', 'misogynist']

import re
import string
import nltk
from nltk.stem import WordNetLemmatizer 

# nltk.download('stopwords') #only if you have not downloaded the stopwords of nltk
def preprocess_text(text):
    # remove all punctuation
    text = re.sub(r'[^\w\d\s]', ' ', text)
    # convert to lower case
    text = re.sub(r'^\s+|\s+?$', '', text.lower())    
    # remove superscripts numbers
    #text = re.sub(r'\p{No}', ' ', text)
    # remove other unwanted characters
    text = re.sub(r'\d', '', text)
    text = re.sub('ã', '', text)
    text = re.sub('â', '', text)
    text = re.sub('user', '', text)
    text = re.sub('µ', '', text)
    text = re.sub('¼', '', text)
    text = re.sub('¾', '', text)
    text = re.sub('½', '', text)
    text = re.sub('³', '', text)
    text = re.sub('ª', '', text)
    text = re.sub('º', '', text)
    text = re.sub('¹', '', text)    
    text = re.sub('²', '', text)
    
    # collapse all white spaces
    text = re.sub(r'\s+', ' ', text)
    # remove stop words and perform stemming
    stop_words = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer() 
    return ' '.join(
        lemmatizer.lemmatize(term) 
        for term in text.split()
        if term not in set(stop_words)
    )

df['processed_text'] = df.tweet.apply(lambda row : preprocess_text(row))
df.head()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

tfidf_vec = TfidfVectorizer(ngram_range=(1,2))
tfidf_data = tfidf_vec.fit_transform(df.processed_text)
#tfidf_data = pd.DataFrame(tfidf_data).toarray()
#tfidf_data.head()

X_train, X_test, y_train, y_test = train_test_split(tfidf_data, df['label'], test_size=0.0, random_state = 42)

# Logistic regression
from sklearn.linear_model import LogisticRegression
clf_log = LogisticRegression(random_state=2019, C=10000).fit(X_train, y_train)


# ------------------------Testing-------------------------------

predictions = clf_log.predict(X_test).tolist()
wrong = []
count = 0
for i in range(len(y_test)):
    if y_test.iloc[i] != predictions[i]:
        count += 1
        wrong.append(i)

      
print('Total number of test cases', len(y_test))
print('Number of wrong of predictions', count)

from sklearn.metrics import classification_report
print(classification_report(predictions, y_test))

# ---------------------------------------------------------------
df_test = pd.read_csv('dataset/test_tweets_anuFYb8.csv', encoding='latin-1')
df_test.head()

df_test['processed_text'] = df_test.tweet.apply(lambda row : preprocess_text(row))
df_test.head()

t = tfidf_vec.transform(df_test.processed_text)

df_test["prediction"] = clf_log.predict(t)

submission = pd.read_csv('submission.csv', encoding='latin-1')
submission['id'] = df_test['id']
submission['label'] = df_test['prediction']

df_test['prediction'].value_counts()

pd.DataFrame(submission, columns=['id','label']).to_csv('submission3.csv', index = False)



