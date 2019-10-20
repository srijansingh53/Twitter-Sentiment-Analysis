# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 09:51:00 2019

@author: SRIJAN
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

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

def hate_label(text):
    
    for term in text.split():
        if term in hate_words:
            return 1
        else:
            continue
        

df['hate_label'] = df.processed_text.apply(lambda row : hate_label(row))
df['hate_label'] = df['hate_label'].fillna(0)
df.head()