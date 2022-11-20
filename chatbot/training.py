# file dedicated for training and preprocessing the dataset

import json
from utils import bag_of_words, preprocess
import pickle
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NN

# load the raw dataset (intents)
intents = json.loads(open('/Users/nick/Desktop/Codejam/ChatbotMcGIll/chatbot/intents.json').read())

# collect all the patterns, and tags, divide them up 
all_words, classes, doc = preprocess(intents)
dataset = []

# we need the actual dataset to be numerical 
# so they need to be vectorized in BagOfWords - preprocessing part
for (tk_word_list, tag) in doc:
    bow = bag_of_words(tk_word_list, all_words)

    y_i = np.zeros(len(classes))
    # locate the index of tag
    label = classes.index(tag)
    # y_i[label] = 1
    dataset.append([bow, label])
    
# shuffle dataset & turn into np.array
random.shuffle(dataset)
dataset = np.array(dataset)

X_train = dataset[:, 0]
y_train = dataset[:, 1]

class ChatDataSet(Dataset):
    def __init__(self):
        self.N = len(X_train)
        self.X_data = X_train
        self.y_data = y_train
        # hyperparameters for training
        self.batch_size = 8
        self.hidden_size = 8
        self.num_epochs = 700
        self.input_size = len(X_train[0])
        self.output_size = len(classes)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    # returns sample size
    def __len__(self):
        return self.N


# fitting the model

# loading the dataset
dataset_preprocessed = ChatDataSet()
train_loader = DataLoader(dataset=dataset_preprocessed, batch_size=dataset_preprocessed.batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#fit the dataset
model = NN(dataset_preprocessed.input_size, dataset_preprocessed.hidden_size, dataset_preprocessed.output_size).to(device)

# loss and optimizer
loss_function = nn.CrossEntropyLoss() # CE Loss function
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_all  = []
for epoch in range(dataset_preprocessed.num_epochs):
    for (words, labels) in train_loader:
        # forward
        # fitting the dataset 
        y_pred = model(words)        
        ls = loss_function(y_pred, labels)
        # backward and optimizer step
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()
        # printing the entropy loss
        if (epoch +1) % 100 == 0:
            loss_all.append(ls.item())
            print(f'epoch {epoch+1}/{(dataset_preprocessed.num_epochs)}, loss = {ls.item(): .4f}')

# save data as dict
data = {
    "model_state" : model.state_dict(),
    "all_words" : all_words,
    "classes" : classes,
    "input_size" : dataset_preprocessed.input_size,
    "output_size": dataset_preprocessed.output_size,
    "hidden_size": dataset_preprocessed.hidden_size
}

# saving the data model
torch.save(data, "data.pth")