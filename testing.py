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

