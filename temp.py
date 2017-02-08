import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer


#df stores the training input
df = pd.read_csv('train_input.csv')
df['taggedText'] = ""
df['taggedText'] = df['taggedText'].astype(object)
#df2 stores the training labels
df2 = pd.read_csv('train_output.csv')

hockey = []
movies = []
nba = []
news = []
nfl = []
politics = []
soccer = []
worldnews = []

#for all conversations in the training set, separate them into lists according to category
for convo in range(len(df2.index)):
    if df2.loc[convo, 'category'] == 'hockey':
        hockey.append(df.loc[convo, 'conversation'])
    elif df2.loc[convo, 'category'] == 'movies':
        movies.append(df.loc[convo, 'conversation'])
    elif df2.loc[convo, 'category'] == 'nba':
        nba.append(df.loc[convo, 'conversation'])
    elif df2.loc[convo, 'category'] == 'news':
        news.append(df.loc[convo, 'conversation'])
    elif df2.loc[convo, 'category'] == 'nfl':
        nfl.append(df.loc[convo, 'conversation'])
    elif df2.loc[convo, 'category'] == 'politics':
        politics.append(df.loc[convo, 'conversation'])
    elif df2.loc[convo, 'category'] == 'soccer':
        soccer.append(df.loc[convo, 'conversation'])
    elif df2.loc[convo, 'category'] == 'worldnews':
        worldnews.append(df.loc[convo, 'conversation'])

tokenizer = RegexpTokenizer(r'\w+')

#get the frequency distribution of nouns and verbs for each category
hockey = ' '.join(hockey)
hockText = tokenizer.tokenize(hockey)
hockTagText = nltk.pos_tag(hockText, tagset='universal')
words = []
for word, pos in hockTagText:
    if pos == 'NOUN' or pos == 'VERB':
        words.append(word)
hockDist = nltk.FreqDist(words)
hock_df = pd.DataFrame(hockDist, index = [0])
hock_df.to_csv('hock_freq.csv')

#hock_word_tag_fd = nltk.FreqDist(hockTagText)
#hockFreq = [wt[0] for (wt, _) in hock_word_tag_fd.most_common() if wt[1] == 'NOUN']
            
movies = ' '.join(movies)
moviText = tokenizer.tokenize(movies)
moviTagText = nltk.pos_tag(moviText, tagset='universal')
words = []
for word, pos in moviTagText:
    if pos == 'NOUN' or pos == 'VERB':
        words.append(word)
moviDist = nltk.FreqDist(words)
movie_df = pd.DataFrame(moviDist, index = [0])
movie_df.to_csv('movie_freq.csv')
#movi_word_tag_fd = nltk.FreqDist(moviTagText)
#moviFreq = [wt[0] for (wt, _) in movi_word_tag_fd.most_common() if wt[1] == 'NOUN']
            
nba = ' '.join(nba)
nbaText = tokenizer.tokenize(nba)
nbaTagText = nltk.pos_tag(nbaText, tagset='universal')
words = []
for word, pos in nbaTagText:
    if pos == 'NOUN' or pos == 'VERB':
        words.append(word)
nbaDist = nltk.FreqDist(words)
nba_df = pd.DataFrame(nbaDist, index = [0])
nba_df.to_csv('nba_freq.csv')
#nba_word_tag_fd = nltk.FreqDist(nbaTagText)
#nbaFreq = [wt[0] for (wt, _) in nba_word_tag_fd.most_common() if wt[1] == 'NOUN']
           
news = ' '.join(news)
newsText = tokenizer.tokenize(news)
newsTagText = nltk.pos_tag(newsText, tagset='universal')
words = []
for word, pos in newsTagText:
    if pos == 'NOUN' or pos == 'VERB':
        words.append(word)
newsDist = nltk.FreqDist(words)
news_df = pd.DataFrame(newsDist, index = [0])
news_df.to_csv('news_freq.csv')
#news_word_tag_fd = nltk.FreqDist(newsTagText)
#newsFreq = [wt[0] for (wt, _) in news_word_tag_fd.most_common() if wt[1] == 'NOUN']
            
nfl = ' '.join(nfl)
nflText = tokenizer.tokenize(nfl)
nflTagText = nltk.pos_tag(nflText, tagset='universal')
words = []
for word, pos in nflTagText:
    if pos == 'NOUN' or pos == 'VERB':
        words.append(word)
nflDist = nltk.FreqDist(words)
nfl_df = pd.DataFrame(nflDist, index = [0])
nfl_df.to_csv('nfl_freq.csv')
#nfl_word_tag_fd = nltk.FreqDist(nflTagText)
#nflFreq = [wt[0] for (wt, _) in nfl_word_tag_fd.most_common() if wt[1] == 'NOUN']
        
politics = ' '.join(politics)
poliText = tokenizer.tokenize(politics)
poliTagText = nltk.pos_tag(poliText, tagset='universal')
words = []
for word, pos in poliTagText:
    if pos == 'NOUN' or pos == 'VERB':
        words.append(word)
poliDist = nltk.FreqDist(words)
poli_df = pd.DataFrame(poliDist, index = [0])
poli_df.to_csv('poli_freq.csv')
#poli_word_tag_fd = nltk.FreqDist(poliTagText)
#poliFreq = [wt[0] for (wt, _) in poli_word_tag_fd.most_common() if wt[1] == 'NOUN']
           
soccer = ' '.join(soccer)
soccText = tokenizer.tokenize(soccer)
soccTagText = nltk.pos_tag(soccText, tagset='universal')
words = []
for word, pos in soccTagText:
    if pos == 'NOUN' or pos == 'VERB':
        words.append(word)
soccDist = nltk.FreqDist(words)
socc_df = pd.DataFrame(soccDist, index = [0])
socc_df.to_csv('socc_freq.csv')
#socc_word_tag_fd = nltk.FreqDist(soccTagText)
#soccFreq = [wt[0] for (wt, _) in socc_word_tag_fd.most_common() if wt[1] == 'NOUN']
            
worldnews = ' '.join(worldnews)
worldText = tokenizer.tokenize(worldnews)
worldTagText = nltk.pos_tag(worldText, tagset='universal')
words = []
for word, pos in worldTagText:
    if pos == 'NOUN' or pos == 'VERB':
        words.append(word)
worldDist = nltk.FreqDist(words)
world_df = pd.DataFrame(worldDist, index = [0])
world_df.to_csv('world_freq.csv')
#world_word_tag_fd = nltk.FreqDist(worldTagText)
#worldFreq = [wt[0] for (wt, _) in world_word_tag_fd.most_common() if wt[1] == 'NOUN']
             

        
#for conversation in range(10):
#    text = nltk.word_tokenize(df.loc[conversation,'conversation'])
#    word_tag_fd = nltk.FreqDist(nltk.pos_tag(text, tagset='universal'))
#    df.set_value(conversation,'taggedText',[wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == 'NOUN'])


#TODO: normalize the frequency distribution ?
#TODO: save the frequency distribution onto csv files (so we don't have to rerun)
#TODO: load the csv files into df's and combine them into a np matrix
#TODO: turn the matrix data into TF-IDF using scikit learn
#TODO: use the top N features for Naive Bayes (cross validation techniques, ROC curve)
