#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:04:32 2017

@author: reidkostenuk
"""

import nltk
import pandas as pd
import numpy as np
import time

startTime = time.time()

dfTI = pd.read_csv('train_input.csv')
dfTO = pd.read_csv('train_output.csv')

frames = [dfTI, dfTO]
df = pd.concat(frames, axis=1)
df.drop(df.columns[2], axis=1, inplace=True)

hockey = []
movies = []
nba = []
news = []
nfl = [] 
politics = []
soccer = [] 
worldnews = []

tokenizer = nltk.RegexpTokenizer(r'\w+')
word_lemma = nltk.WordNetLemmatizer()
cachedStopWords = set(nltk.corpus.stopwords.words('english'))
cachedStopWords.update(('number', 'com', 'speaker_1', 'speaker_2', 'speaker_3', 'speaker_4', 'speaker_5', 'speaker_6', 'speaker_7', 'speaker_8', 'speaker_9', 'speaker_10'))

for i in range(len(df)):
    token = tokenizer.tokenize(df.loc[i, 'conversation'])
    token = [word for word in token if word not in cachedStopWords]
    tokenLem = [None]*len(token)
    for w in range(len(token)):
        tokenLem[w] = word_lemma.lemmatize(token[w])
    freq = nltk.FreqDist(tokenLem)
    
    if df.loc[i,'category'] == 'hockey':
        hockey.extend(freq.keys())
    elif df.loc[i,'category'] == 'movies':
        movies.extend(freq.keys())
    elif df.loc[i,'category'] == 'nba':
        nba.extend(freq.keys())
    elif df.loc[i,'category'] == 'news':
        news.extend(freq.keys())
    elif df.loc[i,'category'] == 'nfl':
        nfl.extend(freq.keys())
    elif df.loc[i,'category'] == 'politics':
        politics.extend(freq.keys())
    elif df.loc[i,'category'] == 'soccer':
        soccer.extend(freq.keys())
    elif df.loc[i,'category'] == 'worldnews':
        worldnews.extend(freq.keys())
    
hockeyTag = nltk.pos_tag(hockey, tagset='universal')
hockeyNV = []
for word, pos in hockeyTag:
    if pos == 'NOUN':# or pos == 'VERB':
        hockeyNV.append(word)
hockeyDist = nltk.FreqDist(hockeyNV)
hockeyWord = []
hockeyProb = []
for key, value in hockeyDist.items():
    total = len(hockeyDist)
    hockeyWord.append(key)
    hockeyProb.append(float(value)/total)
hockeyWord = np.asarray(hockeyWord)
hockeyProb = np.asarray(hockeyProb)
hockeyWordPd = pd.DataFrame(hockeyWord)
hockeyProbPd = pd.DataFrame(hockeyProb)
hockeyFrame = [hockeyWordPd, hockeyProbPd]
dfHockey = pd.concat(hockeyFrame, axis=1)
dfHockey.columns = ['word', 'probability']
dfHockey = dfHockey.sort_values(by='probability', ascending=0)

moviesTag = nltk.pos_tag(movies, tagset='universal')
moviesNV = []
for word, pos in moviesTag:
    if pos == 'NOUN':# or pos == 'VERB':
        moviesNV.append(word)
moviesDist = nltk.FreqDist(moviesNV)
moviesWord = []
moviesProb = []
for key, value in moviesDist.items():
    total = len(moviesDist)
    moviesWord.append(key)
    moviesProb.append(float(value)/total)
moviesWord = np.asarray(moviesWord)
moviesProb = np.asarray(moviesProb)
moviesWordPd = pd.DataFrame(moviesWord)
moviesProbPd = pd.DataFrame(moviesProb)
moviesFrame = [moviesWordPd, moviesProbPd]
dfMovies = pd.concat(moviesFrame, axis=1)
dfMovies.columns = ['word', 'probability']
dfMovies = dfMovies.sort_values(by='probability', ascending=0)

nbaTag = nltk.pos_tag(nba, tagset='universal')
nbaNV = []
for word, pos in nbaTag:
    if pos == 'NOUN':# or pos == 'VERB':
        nbaNV.append(word)
nbaDist = nltk.FreqDist(nbaNV)
nbaWord = []
nbaProb = []
for key, value in nbaDist.items():
    total = len(nbaDist)
    nbaWord.append(key)
    nbaProb.append(float(value)/total)
nbaWord = np.asarray(nbaWord)
nbaProb = np.asarray(nbaProb)
nbaWordPd = pd.DataFrame(nbaWord)
nbaProbPd = pd.DataFrame(nbaProb)
nbaFrame = [nbaWordPd, nbaProbPd]
dfNba = pd.concat(nbaFrame, axis=1)
dfNba.columns = ['word', 'probability']
dfNba = dfNba.sort_values(by='probability', ascending=0)

newsTag = nltk.pos_tag(news, tagset='universal')
newsNV = []
for word, pos in newsTag:
    if pos == 'NOUN':# or pos == 'VERB':
        newsNV.append(word)
newsDist = nltk.FreqDist(newsNV)
newsWord = []
newsProb = []
for key, value in newsDist.items():
    total = len(newsDist)
    newsWord.append(key)
    newsProb.append(float(value)/total)
newsWord = np.asarray(newsWord)
newsProb = np.asarray(newsProb)
newsWordPd = pd.DataFrame(newsWord)
newsProbPd = pd.DataFrame(newsProb)
newsFrame = [newsWordPd, newsProbPd]
dfNews = pd.concat(newsFrame, axis=1)
dfNews.columns = ['word', 'probability']
dfNews = dfNews.sort_values(by='probability', ascending=0)

nflTag = nltk.pos_tag(nfl, tagset='universal')
nflNV = []
for word, pos in nflTag:
    if pos == 'NOUN':# or pos == 'VERB':
        nflNV.append(word)
nflDist = nltk.FreqDist(nflNV)
nflWord = []
nflProb = []
for key, value in nflDist.items():
    total = len(hockeyDist)
    nflWord.append(key)
    nflProb.append(float(value)/total)
nflWord = np.asarray(nflWord)
nflProb = np.asarray(nflProb)
nflWordPd = pd.DataFrame(nflWord)
nflProbPd = pd.DataFrame(nflProb)
nflFrame = [nflWordPd, nflProbPd]
dfNfl = pd.concat(nflFrame, axis=1)
dfNfl.columns = ['word', 'probability']
dfNfl = dfNfl.sort_values(by='probability', ascending=0)

politicsTag = nltk.pos_tag(politics, tagset='universal')
politicsNV = []
for word, pos in politicsTag:
    if pos == 'NOUN':# or pos == 'VERB':
        politicsNV.append(word)
politicsDist = nltk.FreqDist(politicsNV)
politicsWord = []
politicsProb = []
for key, value in politicsDist.items():
    total = len(politicsDist)
    politicsWord.append(key)
    politicsProb.append(float(value)/total)
politicsWord = np.asarray(politicsWord)
politicsProb = np.asarray(politicsProb)
politicsWordPd = pd.DataFrame(politicsWord)
politicsProbPd = pd.DataFrame(politicsProb)
politicsFrame = [politicsWordPd, politicsProbPd]
dfPolitics = pd.concat(politicsFrame, axis=1)
dfPolitics.columns = ['word', 'probability']
dfPolitics = dfPolitics.sort_values(by='probability', ascending=0)

soccerTag = nltk.pos_tag(soccer, tagset='universal')
soccerNV = []
for word, pos in soccerTag:
    if pos == 'NOUN':# or pos == 'VERB':
        soccerNV.append(word)
soccerDist = nltk.FreqDist(soccerNV)
soccerWord = []
soccerProb = []
for key, value in soccerDist.items():
    total = len(soccerDist)
    soccerWord.append(key)
    soccerProb.append(float(value)/total)
soccerWord = np.asarray(soccerWord)
soccerProb = np.asarray(soccerProb)
soccerWordPd = pd.DataFrame(soccerWord)
soccerProbPd = pd.DataFrame(soccerProb)
soccerFrame = [soccerWordPd, soccerProbPd]
dfSoccer = pd.concat(soccerFrame, axis=1)
dfSoccer.columns = ['word', 'probability']
dfSoccer = dfSoccer.sort_values(by='probability', ascending=0)

worldnewsTag = nltk.pos_tag(worldnews, tagset='universal')
worldnewsNV = []
for word, pos in worldnewsTag:
    if pos == 'NOUN':# or pos == 'VERB':
        worldnewsNV.append(word)
worldnewsDist = nltk.FreqDist(worldnewsNV)
worldnewsWord = []
worldnewsProb = []
for key, value in worldnewsDist.items():
    total = len(worldnewsDist)
    worldnewsWord.append(key)
    worldnewsProb.append(float(value)/total)
worldnewsWord = np.asarray(worldnewsWord)
worldnewsProb = np.asarray(worldnewsProb)
worldnewsWordPd = pd.DataFrame(worldnewsWord)
worldnewsProbPd = pd.DataFrame(worldnewsProb)
worldnewsFrame = [worldnewsWordPd, worldnewsProbPd]
dfWorldnews = pd.concat(worldnewsFrame, axis=1)
dfWorldnews.columns = ['word', 'probability']
dfWorldnews = dfWorldnews.sort_values(by='probability', ascending=0)

dfHockey.to_csv('condProbHockey.csv')
dfMovies.to_csv('condProbMovies.csv')
dfNba.to_csv('condProbNba.csv')
dfNews.to_csv('condProbNews.csv')
dfPolitics.to_csv('condProbPolitics.csv')
dfNfl.to_csv('condProbNfl.csv')
dfSoccer.to_csv('condProbSoccer.csv')
dfWorldnews.to_csv('condProbWorldnews.csv')

stopTime = time.time()
print("--- %s seconds ---" % (stopTime - startTime))