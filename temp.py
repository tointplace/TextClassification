import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

df = pd.read_csv('train_input.csv')
df['taggedText'] = ""
df['taggedText'] = df['taggedText'].astype(object)
df2 = pd.read_csv('train_output.csv')

hockey = []
movies = []
nba = []
news = []
nfl = []
politics = []
soccer = []
worldnews = []

for convo in range(50):
    if df2.loc[convo,'category'] == 'hockey':
        hockey.append(df.loc[convo,'conversation'])
    elif df2.loc[convo,'category'] == 'movies':
        movies.append(df.loc[convo,'conversation'])
    elif df2.loc[convo,'category'] == 'nba':
        nba.append(df.loc[convo,'conversation'])
    elif df2.loc[convo,'category'] == 'news':
        news.append(df.loc[convo,'conversation'])
    elif df2.loc[convo,'category'] == 'nfl':
        nfl.append(df.loc[convo,'conversation'])
    elif df2.loc[convo,'category'] == 'politics':
        politics.append(df.loc[convo,'conversation'])
    elif df2.loc[convo,'category'] == 'soccer':
        soccer.append(df.loc[convo,'conversation'])
    elif df2.loc[convo,'category'] == 'worldnews':
        worldnews.append(df.loc[convo,'conversation'])

tokenizer = RegexpTokenizer(r'\w+')
        
hockey = ' '.join(hockey)
hockText = tokenizer.tokenize(hockey)
#hockText = nltk.word_tokenize(hockey)
hockTagText = nltk.pos_tag(hockText, tagset='universal')
hock_word_tag_fd = nltk.FreqDist(hockTagText)
hockFreq = [wt[0] for (wt, _) in hock_word_tag_fd.most_common() if wt[1] == 'NOUN']
            
movies = ' '.join(movies)
moviText = tokenizer.tokenize(movies)
#moviText = nltk.word_tokenize(movies)
moviTagText = nltk.pos_tag(moviText, tagset='universal')
movi_word_tag_fd = nltk.FreqDist(moviTagText)
moviFreq = [wt[0] for (wt, _) in movi_word_tag_fd.most_common() if wt[1] == 'NOUN']
            
nba = ' '.join(nba)
nbaText = tokenizer.tokenize(nba)
#nbaText = nltk.word_tokenize(nba)
nbaTagText = nltk.pos_tag(nbaText, tagset='universal')
nba_word_tag_fd = nltk.FreqDist(nbaTagText)
nbaFreq = [wt[0] for (wt, _) in nba_word_tag_fd.most_common() if wt[1] == 'NOUN']
           
news = ' '.join(news)
newsText = tokenizer.tokenize(news)
#newsText = nltk.word_tokenize(news)
newsTagText = nltk.pos_tag(newsText, tagset='universal')
news_word_tag_fd = nltk.FreqDist(newsTagText)
newsFreq = [wt[0] for (wt, _) in news_word_tag_fd.most_common() if wt[1] == 'NOUN']
            
nfl = ' '.join(nfl)
nflText = tokenizer.tokenize(nfl)
#nflText = nltk.word_tokenize(nfl)
nflTagText = nltk.pos_tag(nflText, tagset='universal')
nfl_word_tag_fd = nltk.FreqDist(nflTagText)
nflFreq = [wt[0] for (wt, _) in nfl_word_tag_fd.most_common() if wt[1] == 'NOUN']
        
politics = ' '.join(politics)
poliText = tokenizer.tokenize(politics)
#poliText = nltk.word_tokenize(politics)
poliTagText = nltk.pos_tag(poliText, tagset='universal')
poli_word_tag_fd = nltk.FreqDist(poliTagText)
poliFreq = [wt[0] for (wt, _) in poli_word_tag_fd.most_common() if wt[1] == 'NOUN']
           
soccer = ' '.join(soccer)
soccText = tokenizer.tokenize(soccer)
#soccText = nltk.word_tokenize(soccer)
soccTagText = nltk.pos_tag(soccText, tagset='universal')
socc_word_tag_fd = nltk.FreqDist(soccTagText)
soccFreq = [wt[0] for (wt, _) in socc_word_tag_fd.most_common() if wt[1] == 'NOUN']
            
worldnews = ' '.join(worldnews)
worldText = tokenizer.tokenize(worldnews)
#worldText = nltk.word_tokenize(worldnews)
worldTagText = nltk.pos_tag(worldText, tagset='universal')
world_word_tag_fd = nltk.FreqDist(worldTagText)
worldFreq = [wt[0] for (wt, _) in world_word_tag_fd.most_common() if wt[1] == 'NOUN']
             

        
#for conversation in range(10):
#    text = nltk.word_tokenize(df.loc[conversation,'conversation'])
#    word_tag_fd = nltk.FreqDist(nltk.pos_tag(text, tagset='universal'))
#    df.set_value(conversation,'taggedText',[wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == 'NOUN'])
    