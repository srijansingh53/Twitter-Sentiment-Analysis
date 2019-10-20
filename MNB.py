# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:54:24 2019

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

tfidf_vec = TfidfVectorizer(ngram_range=(1,2))
tfidf_data = tfidf_vec.fit_transform(df.processed_text)
#tfidf_data = pd.DataFrame(tfidf_data)
#tfidf_data.head()

X_train, X_test, y_train, y_test = train_test_split(tfidf_data, df['label'], test_size=0.0, random_state = 42)

spam_filter = MultinomialNB(alpha=0.2)
spam_filter.fit(X_train, y_train)







"""

# ------------------------Testing-------------------------------"

predictions = spam_filter.predict(X_test).tolist()
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

text = "yoyas market deming nm sell beer white said closed hispanic walked n sold"

text = [preprocess_text(text)]
t = tfidf_vec.transform(text)

spam_filter.predict(t)[0]

"""






