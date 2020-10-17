import pandas as pd
import numpy as np
import sys
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import keras
import pickle
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

#Load data from command line
file_to_open=sys.argv[1] 
df=pd.read_fwf(file_to_open, sep=" ", header=None) #Assuming that the data is a txt file as it was provided this way
df.columns = ["Text"]

#Load previously saved categories and convert them to a list
categories = pd.read_csv("categories.csv")
categories=categories.iloc[:,0].tolist()

#Specify max number of words for the tokenizer
max_words = 7000

#Specify number of categories
num_classes = len(categories)

#loading previously fitted Tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#encode into sequences
text_tokenized = tokenizer.texts_to_sequences(list(df.Text))
text_tokenized = tokenizer.sequences_to_matrix(text_tokenized, mode='tfidf')

#Model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#Load pre-trained model
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    loaded_model = load_model('model.h5',compile=False)

#Create a column to hold predictions
df["prediction"] = np.argmax(loaded_model.predict(text_tokenized), axis=1)

#save predictions
df.to_csv("testSet.csv", index=False, header=False)

print("Prediction completed")