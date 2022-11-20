# natural language tool kit
import nltk 

#lemmatizer function 
lemmatizer = nltk.stem.WordNetLemmatizer()
import numpy as np

def preprocess(data):

    ignore_letters = ['?', '!', '.', ',']
    all_words = []
    classes = []
    documents = []
    # intents are our entire dataset
# patterns contain all the list of sentences we might expect for each tag/scenario

    for d in data['intents']:
        for pattern in d['patterns']:
            tk_word_list = tokenize(pattern)
            # do not want to make 2d arrays -> hence, extend
            all_words.extend(tk_word_list)
            documents.append((tk_word_list, d['tag']) )

            # create an array for tags (label)
            if d['tag'] not in classes:
                classes.append(d['tag'])

    all_words = [lemmatize(word) for word in all_words if word not in ignore_letters] 

    # get rid of duplicates
    all_words = sorted(set(all_words))
    classes = sorted(set(classes)) 
    return all_words, classes, documents



# tokenizer for training : splits up a sentence into separate words
def tokenize(pattern): return nltk.word_tokenize(pattern) 

# stems the words 
def lemmatize(word): return lemmatizer.lemmatize(word.lower())


def bag_of_words(tokenized_pattern, all_words):
    bow = np.zeros(len(all_words), dtype=np.float32)
    tokenized_pattern = [lemmatize(word) for word in tokenized_pattern]
    for idx, w in enumerate(all_words):
        if w in tokenized_pattern:
            bow[idx] = 1.0
    return bow

