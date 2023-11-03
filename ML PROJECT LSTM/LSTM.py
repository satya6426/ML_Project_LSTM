import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding


data = pd.read_csv('shakespeare_data.csv')  

# Preprocessing the text data
corpus = " ".join(data['text'])  
corpus = corpus.lower()  
chars = sorted(list(set(corpus)))
char_indices = {char: idx for idx, char in enumerate(chars)}
indices_char = {idx: char for idx, char in enumerate(chars)}


maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(corpus) - maxlen, step):
    sentences.append(corpus[i: i + maxlen])
    next_chars.append(corpus[i + maxlen])


x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Building the LSTM model
model = Sequential([
    LSTM(128, input_shape=(maxlen, len(chars))),
    Dense(len(chars), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(x, y, batch_size=128, epochs=30)

# Function to generate text
def generate_text(seed_text, next_words=100):
    for _ in range(next_words):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(seed_text):
            x_pred[0, t, char_indices[char]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.random.choice(len(chars), p=preds)
        next_char = indices_char[next_index]
        seed_text = seed_text[1:] + next_char
        print(next_char, end='')

# Generate text using a seed sentence
seed_sentence = "To be or not to be"
generate_text(seed_sentence)
