import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def lemmatize(word):
    return lemmatizer.lemmatize(word.lower())

def bag_of_words(tokenized_sentence, words):
    tokenized_sentence = [lemmatize(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)

    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1

    return bag