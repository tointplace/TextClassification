import nltk
import numpy as np
import pandas as pd

df = pd.read_csv('train_input.csv')
df['taggedText'] = ""
df['taggedText'] = df['taggedText'].astype(object)
#df2 = pd.DataFrame

for conversation in range(5):
    text = nltk.word_tokenize(df.loc[conversation,'conversation'])
    word_tag_fd = nltk.FreqDist(nltk.pos_tag(text, tagset='universal'))
    df.set_value(conversation,'taggedText',[wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == 'NOUN'])
    