# file dedicated for training and preprocessing the dataset

import json
from nltk_utils import tokenize, lemmatize, bag_of_words
import pickle
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NN

# load the dataset (intents)
intents = json.loads(open('/Users/nick/Desktop/Codejam/ChatbotMcGIll/chatbot/intents.json').read())

all_words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']


# intents are our entire dataset
# patterns contain all the list of sentences we might expect for each tag/scenario

for intent in intents['intents']:
    for pattern in intent['patterns']:
        tk_word_list = tokenize(pattern)
        # do not want to make 2d arrays -> hence, extend
        all_words.extend(tk_word_list)
        documents.append((tk_word_list, intent['tag']) )

        # create an array for tags (label)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

all_words = [lemmatize(word) for word in all_words if word not in ignore_letters] 

# get rid of duplicates
all_words = sorted(set(all_words))
classes = sorted(set(classes)) 

X = []
y = []
dataset = []

# we need the actual dataset to be numerical 
# so they need to be vectorized in BagOfWords - preprocessing part
for (tk_word_list, tag) in documents:
    bow = bag_of_words(tk_word_list, all_words)
    X.append(bow)

    y_i = np.zeros(len(classes))
    # locate the index of tag
    label = classes.index(tag)
    y_i[label] = 1
    y.append(y_i)

    dataset.append([bow, y_i])
    
# shuffle dataset & turn into np.array
random.shuffle(dataset)
dataset = np.array(dataset)
# turn this into np array
X_train = np.array(X)
y_train = np.array(y)


class ChatDataSet(Dataset):
    def __init__(self):
        self.N = len(X_train)
        self.X_data = X_train
        self.y_data = y_train
        # hyperparameters for training
        self.batch_size = 8
        self.hidden_size = 8

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    # returns sample size
    def __len__(self):
        return self.N


# #hyperparameters
batch_size = 8
hidden_size = 8
input_size = len(X_train[0])
output_size = len(classes)

print(input_size, len(all_words))
print(output_size, classes)
num_epochs = 700


dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=dataset.batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NN(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss() # CE Loss function
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        print(words, labels)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # printing the entropy loss
        if (epoch +1) % 100 == 0:
            print(f'epoch {epoch+1}/{(num_epochs)}, loss = {loss.item(): .4f}')

print(f'final loss, loss ={loss.item(): .4f}') 


data = {
    "model_state" : model.state_dict(),
    "input_size" : input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words" : all_words,
    "classes" : classes,
}

FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete. File saved to {FILE}')
