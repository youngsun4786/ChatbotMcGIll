import random 
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
intents = json.loads(open('/Users/nick/Desktop/Codejam/ChatbotMcGIll/chatbot/intents.json').read())

FILE = "data.pth"
data = torch.load(FILE)
input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
classes = data['classes']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
# knows our learned parameters
model.load_state_dict(model_state)
model.eval()

bot_name = "NAAS"

print("Let's chat! type 'quit' to exit...")

while True:
    sentence = input ('You: ')
    if sentence == 'quit':
        break
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    # 1 sample
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    # gives our predictions
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = classes[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # if the probability is high enough
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f'{bot_name} {random.choice(intent["responses"])}')
    else:
        print(f'{bot_name} : I do not understand ...')



