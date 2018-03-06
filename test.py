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

