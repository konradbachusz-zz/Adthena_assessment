import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn import metrics
import keras
from keras.preprocessing.text import Tokenizer
from collections import Counter
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


#Load training data from csv from command line
file_to_open=sys.argv[1]
df = pd.read_csv(file_to_open, sep=',', names=['Text', 'category_id'])

#Remove stop words
from nltk.corpus import stopwords
stop = set(stopwords.words("english"))
def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)

df["Text"] = df["Text"].map(remove_stopwords)

# Count unique words
def counter_word(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count

text = df.Text
counter = counter_word(text)

#Separate features from target variable
X=df.Text
y = df.category_id

#Get and save the categories. They will be later used for making predictions
categories = df.category_id.unique()
pd.DataFrame(categories).to_csv("categories.csv", index=False)

#Free up memory
df = None

#Assign number of classes
num_classes = len(categories)

#Convert target variable into a categorical matrix
y = keras.utils.to_categorical(y, num_classes)

#Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Indicate maximum words for the tokenizer. 
#Ideally it is equal to len(counter) but due to limited RAM on my laptop I had to reduce it
max_words = 7000

#Define the tokenizer
tokenizer = Tokenizer(num_words=max_words)

#Fit on to texts
tokenizer.fit_on_texts(X_train)

#Save fitted tokenizer. Will be later used in prediction
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Convert feature series to lists
X_train = list(X_train)
X_test = list(X_test)

#Encode into sequences
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

#Convert feature sequences into matricies and obtain term frequency–inverse document frequency values
X_train = tokenizer.sequences_to_matrix(X_train, mode='tfidf')
X_test = tokenizer.sequences_to_matrix(X_test, mode='tfidf')

#Model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.metrics_names)
batch_size = 320 
epochs = 3

#Fit the model
#Testing: The model displays validation accuracy and loss at each epoch.
history = model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs, verbose=1, validation_split=0.1)

#Display final test loss and accuracy
score = model.evaluate(X_test, y_test,batch_size=batch_size,verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Save the model
model.save("model.h5")

#Notification in the command line
print("Model training complete")