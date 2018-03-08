import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import random
import json
import tflearn
import tensorflow as tf

with open('intents.json') as json_data:
    intents = json.load(json_data)

# Restore training data from pickle
import pickle
data = pickle.load( open('training_data', 'rb'))

train_x = data['train_x']
train_y = data['train_y']
words = data['words']
classes = data['classes']

# Rebuild neural network
net = tflearn.input_data(shape = [None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation = 'softmax')
tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

model.load('./model.tflearn')

# Tokenize and stem all sentences
def cleanup_sentence(sentence):
    # Tokenize sentence
    sentence_words = nltk.word_tokenize(sentence)
    # Reduce all words to stems
    sentence_words = [stemmer.stem(w) for w in sentence_words]
    return sentence_words

# Create bag of words from sentences
def bow(sentence, words, show_details=False):
    # tokenize and stem the input
    sentence_words = cleanup_sentence(sentence)
    # create bag of words
    # We are using a slightly different approach than setup here
    # presumably to allow show_details
    bag = [0]*len(words)
    # Loop over all words in input sentence
    for sw in sentence_words:
        # Then loop over all words in corpus
        for i, w in enumerate(words):
            # If matches, set appropriate bag entry to 1
            if w == sw:
                bag[i] = 1
                if show_details:
                    print("Found word %s in bag " % sw)
    # Return numpy array
    return np.array(bag)
    
# p = bow('is you shop open today?', words)
# print(model.predict([p]))
# print(p)
# print(classes)

# Classification starts
# Placeholder for storing context
context = {}

ERROR_THRESHOLD=0.25
def classify(sentence):
    # Store probability results of model prediction
    results = model.predict([bow(sentence, words)])[0]
    for i, r in enumerate(results):
        print(classes[i], r)
    # Filter out predictions below a level
    results = [[i,r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # Sort by probability desc
    results.sort(key = lambda x : x[1], reverse=True)
    # Initialize return list
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # Return tuple of intent and probability
    return return_list

print(classify("Is your shop open today?"))