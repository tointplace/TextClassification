import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

training_data = pd.read_csv('training_data_1.csv')
word_data = pd.read_csv('entropy_1000.csv')

word_list = []
word_list.extend(word_data['0'])
word_list.extend(word_data['3'])
word_list.extend(word_data['6'])
word_list.extend(word_data['9'])
word_list.extend(word_data['12'])
word_list.extend(word_data['15'])
word_list.extend(word_data['18'])
word_list.extend(word_data['21'])
word_list = list(set(word_list))

tokenizer = RegexpTokenizer(r'\w+')
word_lemma = nltk.WordNetLemmatizer()
cachedStopWords = set(nltk.corpus.stopwords.words('english'))
cachedStopWords.update(('number', 'com', 'speaker_1', 'speaker_2', 'speaker_3', 'speaker_4', 'speaker_5', 'speaker_6', 'speaker_7', 'speaker_8', 'speaker_9', 'speaker_10'))

tf_df = pd.DataFrame(columns=word_list, index = range(len(training_data)))

avg_words = 0
for i in range(len(training_data)):
    if i%1000 == 0:
        print ("iteration", i)
    tokenized_conv = tokenizer.tokenize(training_data.loc[i, 'conversation'])
    tokenized_conv = [word for word in tokenized_conv if word not in cachedStopWords]
    tagText = nltk.pos_tag(tokenized_conv, tagset='universal')
    words = []
    n_words = 0
    for word, pos in tagText:
        if pos == 'NOUN' or pos == 'VERB':
            words.append(word_lemma.lemmatize(word))

    freq = nltk.FreqDist(words)
    tf_entry = pd.DataFrame(columns=word_list)
    for word in word_list:
        if word in freq:
            tf_df.loc[i, word] = freq[word]
            n_words += 1
        else:
            tf_df.loc[i, word] = 0

    avg_words += n_words

avg_words = avg_words/len(training_data)
print(avg_words)
#after reading all conversations
tf_df.to_csv('feature_set_1.csv')