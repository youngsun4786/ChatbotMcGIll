import json
from nltk_utils import tokenize, lemmatize, bag_of_words

import pickle
import numpy as np
import random

# #tensor flow and keras
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import SGD

# load the dataset (intents)
intents = json.loads(
    open('/Users/nick/Desktop/Codejam/ChatbotMcGIll/chatbot/intents.json').read())

all_words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']


# intents are our entire dataset
# patterns contain all the list of sentences we might expect for each tag/scenario

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = tokenize(pattern)
        # do not want to make 2d arrays -> hence, extend
        all_words.extend(word_list)
        documents.append((word_list, intent['tag']))

        # create an array for tags (label)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

all_words = [lemmatize(word)
             for word in all_words if word not in ignore_letters]

# get rid of duplicates
all_words = sorted(set(all_words))  # X (input)
classes = sorted(set(classes))  # y (label)

# create pickle
pickle.dump(all_words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

X_train = []
y_train = []

# we need the actual dataset to be numerical
# so they need to be vectorized in BagOfWords
for (word_list, tag) in documents:
    bow = bag_of_words(word_list, all_words)
    X_train.append(bow)

    # locate the index of tag
    label = classes.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
# testing
print(X_train)
print(y_train)
