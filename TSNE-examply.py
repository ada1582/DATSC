#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 21:40:28 2020

@author: daniyalusmani1
"""


import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def generateDummyData():

    n = 100
    data = {'temperature': np.random.normal(14, 3, n),
        'moisture': np.random.normal(96, 2, n),
        'sunlight': np.random.normal(150, 5, n),
        'color': np.random.choice(['green', 'yellow', 'purple'], 
                                  size=100, 
                                  p=[0.8, 0.1, 0.1])}
    df = pd.DataFrame(data=data)
    df['label'] = df.apply(lambda row: get_label(row['color'],
                                                 row['sunlight'],
                                                 row['temperature']), axis=1)
    print(df.head())
    return df.iloc[:, 0:3].to_numpy(), df.iloc[:, 4].to_numpy()
    

def get_label(color, moisture, temperature):
    if temperature < 10 or temperature > 18:
        return 1
    elif color != 'green':
        return 1
    elif moisture < 94 or moisture > 98:
        return 1
    return 0


#X, y, ids = ut.load_ndvi_uts(cfg.data_path, ut.as_list(2015), cfg.balance_flag)
# X, y, ids = generateDummyData
X, y = generateDummyData()
# https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py

X_embedded = TSNE(n_components=2).fit_transform(X)
print(X_embedded.shape)
print(X[0:5])
print(y[0:5])



