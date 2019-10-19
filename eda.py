# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 09:29:24 2019

@author: SRIJAN
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset/train_E6oV3lV.csv', encoding='latin-1')
df.head()

sns.countplot(df.label)
plt.xlabel('Label')
plt.ylabel('Number of Tweets')

df['length'] = df['tweet'].apply(len)
mpl.rcParams['patch.force_edgecolor'] = True
plt.style.use('seaborn-bright')
df.hist(column='length', by='label', bins=50, figsize=(11,5))

hate_words = ['hate', 'kill', 'racist', 'racism', 'black', 'misogynist']

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









    
