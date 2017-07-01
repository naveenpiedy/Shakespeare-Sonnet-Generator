import numpy as np
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

filename = "ssonnets.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)

print("Total Characters ", n_chars)
print("Total Vocab", n_vocab)
print(chars)

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append([char_to_int[seq_out]])
n_patterns = len(dataX)
print("Total Patterns", n_patterns)

X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)

y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
filename = "weights-improvement-07-1.2043-bigger.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)


int_to_char = dict((i, c) for i, c in enumerate(chars))

start = np.random.randint(0, len(dataX) - 1)
print(len(dataX))

print("Seed:")
#print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
pattern = dataX[start]
#pattern = input("Enter seed: ").lower()
#pattern = list(pattern)
#pattern = [char_to_int[x] for x in pattern]
#pattern = pattern[0:100]
# generate characters
print("Generated Text")

def sample_prediction(prediction, temperature =0.4):
    X = prediction[0]
    a = np.log(X) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
      # sum(X) is approx 1
    #rnd_idx = np.random.choice(len(a), p=a)
    rnd_idx = np.argmax(a)
    return rnd_idx

"""def sample_prediction(char_map, prediction):
    rnd_idx = np.random.choice(len(prediction), p=prediction)
    return char_map[rnd_idx]"""

"""def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))"""


for i in range(300):
    x = np.reshape(pattern, (1, seq_length, 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    #index = sample_prediction(prediction)
    result = int_to_char[index]
    #seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone.")

