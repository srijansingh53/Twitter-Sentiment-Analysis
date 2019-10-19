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

df = pd.read_csv('train_E6oV3lV.csv', encoding='latin-1')
df.head()

df['length'] = df['tweet'].apply(len)
