import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) #1st input layer - maps input nodes -> hidden nodes
        self.l2 = nn.Linear(hidden_size, hidden_size) #2nd (hidden layer) - maps hidden -> hidden
        self.l3 = nn.Linear(hidden_size, num_classes) #3rd (output layer) - maps hidden -> output 
        self.relu = nn.ReLU() # primary activation function relU
        self.relu = nn.ReLU()
    
    # take bag of words 
    # relu - adds complexity to the forward propagation
    def forward(self, X):
        hidden1 = self.l1(X)
        hidden1_applied = self.relu(hidden1)
        hidden2 = self.l2(hidden1_applied)
        hidden2_applied =  self.relu(hidden2)
        pred_label = self.l3(hidden2_applied)
        return pred_label


