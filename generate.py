import numpy as np
import argparse
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

from prep import prep_data
from model import create_lstm_model

ap = argparse.ArgumentParser()
ap.add_argument('--file_path', type=str, default='alice-in-wonderland.txt')
ap.add_argument('--seq_length', type=int, default=100)

args = vars(ap.parse_args())

FILE_PATH = args['file_path']
SEQ_LENGTH = args['seq_length']

raw_text, chars, char_to_int, X, y = prep_data('alice-in-wonderland.txt', SEQ_LENGTH)

n_chars = len(raw_text)
n_vocab = len(chars)
n_patterns = len(X)

model = create_lstm_model(X.shape[1], X.shape[2], y.shape[1])
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
