import json
import pickle
import numpy as np
import random

# natural language tool kit
import nltk 
from nltk.stem import WordNetLemmatizer


#tensor flow and keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.tokenize(pattern) 
        words.append(word_list, intent['tag'])
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


print(documents)
