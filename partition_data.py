import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

training_data = pd.read_csv('train_input.csv')

l = len(training_data)
p = int(l/4)
training_data_1 = training_data[0:p]
training_data_2 = training_data[p:2*p]
training_data_3 = training_data[2*p:3*p]
training_data_4 = training_data[3*p:]

training_data_1.to_csv('training_data_1.csv')
training_data_2.to_csv('training_data_2.csv')
training_data_3.to_csv('training_data_3.csv')
training_data_4.to_csv('training_data_4.csv')

