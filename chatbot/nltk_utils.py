# natural language tool kit
import nltk 
from nltk.stem import WordNetLemmatizer

#stemmer function 
lemmatizer = WordNetLemmatizer()
import numpy as np

# tokenizer for training : splits up a sentence into separate words
def tokenize(pattern): 
    return nltk.word_tokenize(pattern) 

# stems the words 
def lemmatize(word): 
    return lemmatizer.lemmatize(word.lower())

def bag_of_words(tokenized_pattern, all_words):
    tokenized_pattern = [lemmatize(word) for word in tokenized_pattern]
    bow = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_pattern:
            bow[idx] = 1.0
    return bow