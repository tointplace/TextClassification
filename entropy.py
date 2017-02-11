#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 12:13:04 2017

@author: reidkostenuk
"""

import pandas as pd
import numpy as np
import scipy as sp

df_list = []
df_list.append(pd.read_csv('condProbHockey.csv'))
df_list.append(pd.read_csv('condProbMovies.csv'))
df_list.append(pd.read_csv('condProbNba.csv'))
df_list.append(pd.read_csv('condProbNews.csv'))
df_list.append(pd.read_csv('condProbPolitics.csv'))
df_list.append(pd.read_csv('condProbNfl.csv'))
df_list.append(pd.read_csv('condProbSoccer.csv'))
df_list.append(pd.read_csv('condProbWorldnews.csv'))

names = ['hockey', 'movies', 'nba', 'news', 'politics', 'nfl', 'soccer', 'worldnews']
numWords = 200
dfEntropy = pd.DataFrame()
for category in range(len(df_list)):
    tempEnt = []
    for word in range(numWords):
        prob = []
        wordOfInt = df_list[category].loc[word,'word']
        probability = df_list[category].loc[word,'probability']
        for w in range(len(df_list)):
            if wordOfInt in df_list[w]['word'].unique():
                index = np.where(df_list[w]['word'] == wordOfInt)[0][0]
                p = df_list[w].loc[index,'probability']
                prob.append(p)
        ent = sp.stats.entropy(prob)
        tempEnt.append([wordOfInt, ent, probability])
        
    dfTemp = pd.DataFrame(tempEnt)
    #dfTemp = dfTemp.sort_values(by=1)
    
    frames = [dfEntropy,dfTemp]
    dfEntropy = pd.concat(frames, axis=1, ignore_index=True)

dfEntropy.to_csv('entropy.csv')