import random 
import json
import torch
import random as rd
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


# in case the model does not understand any given strings to corpus.
def rand_response(): 
    rand_list=[
        "Please try writing something more descriptive.",
        "Oh! It appears you wrote something I don't understand yet",
        "Do you mind trying to rephrase that?",
        "I'm terribly sorry, I didn't quite catch that.",
        "I can't answer that yet, please try asking something else.",
        "I do not understand...",
        "Beg your pardon?"
    ]

    rand_item = rd.randrange(len(rand_list))
    return rand_list[rand_item]


bot_name = "N.A.A.S"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = classes[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return rand_response()


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
