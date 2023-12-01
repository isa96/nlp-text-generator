import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

data = open('irish-lyrics-eof.txt').read()
corpus = data.lower().split("\n")
# print(corpus[0:5])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) +1

input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_length = max([len(x) for x in input_sequences])
input_pad = pad_sequences(input_sequences, maxlen= max_length, padding='pre')
input_sequences= np.array(input_pad)

x = input_sequences[:,:-1]
y = input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes = total_words)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 64, input_length = max_length-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # tf.keras.layers.GRU(16, dropout=0.5, activation='relu'),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics=["accuracy"])

model.summary()

model.fit(x,y, epochs=100)

def predict(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], padding="pre", maxlen=max_length-1)

        probabilities = model.predict(token_list)
        # predict = np.argmax(probabilities, axis=-1)[0]
        # Pick a random number from [1,2,3]
        choice = np.random.choice([1, 2, 3])

        # Sort the probabilities in ascending order
        # and get the random choice from the end of the array
        predict = np.argsort(probabilities)[0][-choice]

        output_word = tokenizer.index_word[predict]
        seed_text += " " + output_word

    return seed_text

seed_text = "Come all ye maidens young and fair"
next_words = 30

print(predict(seed_text, next_words))