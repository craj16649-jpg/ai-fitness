import json
import random
import pickle
import numpy as np
import nltk

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

from nltk_utils import tokenize, lemmatize, bag_of_words

nltk.download('punkt')
nltk.download('wordnet')

with open('intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']

for intent in intents['intents']:
    tag = intent['tag']
    classes.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        w = [lemmatize(word) for word in w if word not in ignore_letters]
        words.extend(w)
        documents.append((w, tag))

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = bag_of_words(doc[0], words)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:,0]))
train_y = np.array(list(training[:,1]))

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=500, batch_size=8, verbose=1)

model.save('chatbot_model.h5')

print("Model trained successfully!")