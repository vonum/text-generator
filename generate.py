import numpy as np
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load ascii text and covert to lowercase
filename = 'alice-in-wonderland.txt'
raw_text = open(filename).read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text))) # all chars
char_to_int = dict((c, i) for i, c in enumerate(chars)) # map keys => chars, values => ints

n_chars = len(raw_text)
n_vocab = len(chars)
print 'Total Characters: ', n_chars
print 'Total Vocab: ', n_vocab

# prepare the dataset of input to output pairs encoded as integers
# each pattern is comprised of 100 characters and output is the next character
# input = 100 chars, output = 1 char
seq_length = 100
dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
  seq_in = raw_text[i:i + seq_length]
  seq_out = raw_text[i + seq_length]
  dataX.append([char_to_int[char] for char in seq_in])
  dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print 'Total Patterns: ', n_patterns

X = np.reshape(dataX, (n_patterns, seq_length, 1))
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

# load weights
model.load_weights('model.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define oposite mapping
int_to_char = dict((i, c) for i, c in enumerate(chars))

# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print "Seed:"
print "\"", ''.join([int_to_char[value] for value in pattern]), "\""

# generate characters
for i in range(1000):
  x = np.reshape(pattern, (1, len(pattern), 1))
  x = x / float(n_vocab)
  prediction = model.predict(x, verbose=0)
  index = np.argmax(prediction)
  result = int_to_char[index]
  seq_in = [int_to_char[value] for value in pattern]
  sys.stdout.write(result)
  pattern.append(index)
  pattern = pattern[1:len(pattern)]

print "\nDone."
