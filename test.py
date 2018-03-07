import nltk
# Stemmers reduce words to their root (not necessarily valid roots)
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random

import json
# Import json intents file into dict
with open("intents.json") as json_data:
    intents = json.load(json_data)

# Classify words, documents, classification classes

# Initialize everything as empty list
words = []
classes = []
documents = []
ignore_words = ['?']

# Now loop through the intents json dict
# Finding all patterns listed there
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        w = nltk.word_tokenize(pattern)
        # Add the tuple: (words, tag) to the documents in our corpus
        documents.append((w, intent['tag']))
        # Add this to word list
        words.extend(w)
        # Add tag to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        
for w in words:
    print("Words are ",w)

for classs in classes:
    print("Classes are ", classs)

for document in documents:
    print("Document: ", document)
    
# Do some pre-processing. Stemming and lower case and filter out ignore_words
words = [stemmer.stem(w) for w in words if w not in ignore_words]
# Then get uniques, sort it
words = sorted(list(set(words)))
for w in words:
    print("Words are ",w)

# Unique, sorted list of classes
classes = sorted(list(set(classes)))

# Create training data set as a bag of words representing sentences/documents
#  + a one live array of intents
training = []
# This is the initialization of the one-live array
output_empty=[0]*len(classes)

# Iterate through each sentence, creating the bag of words
for doc in documents:
    # Initialize bag of words
    bag=[]
    pattern_words = [stemmer.stem(w) for w in doc[0]]
    # Populate bag of words for each possible word in corpus
    # Turning on where word exists in current doc, and off where it doesn't
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # Take a copy of the original template list for this document
    output_list = list(output_empty)
    # Turn on the index of the current document's tag 
    output_list[classes.index(doc[1])] = 1

    # Append the results to the training list
    training.append([bag, output_list])

# Shuffle list and turn into an np array(what is this?)
random.shuffle(training)
training = np.array(training)

# Create training and testing lists
train_x=list(training[:,0])
train_y=list(training[:,1])

print("First element of training_set x",train_x[0])
print("First element of training_set y",train_y[0])

# This part I Don't understand
# Reset the underlying graph data
tf.reset_default_graph()
# Build neural network
# Input layer
net = tflearn.input_data(shape=[None, len(train_x[0])])
# Next layer
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
# Output layer
net = tflearn.fully_connected(net, len(train_y[0]), activation = 'softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
# Save model
model.save('model.tflearn')


