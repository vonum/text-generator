import numpy as np

from keras.utils import np_utils

def load_text(filename):
  # load ascii text and covert to lowercase
  raw_text = open(filename).read()

  return raw_text.lower()

def reshape(dataX, dataY, seq_length, n_vocab):
  # reshape inputs for neural network
  # reshape X to be [samples, time steps, features]
  n_patterns = len(dataX)
  X = np.reshape(dataX, (n_patterns, seq_length, 1))

  # normalize
  X = X / float(n_vocab)
  # one hot encode the output variable
  y = np_utils.to_categorical(dataY)

  return X, y

def inputs_and_outputs(raw_text, char_to_int, n_chars, seq_length):
  dataX = []
  dataY = []

  # prepare the dataset of input to output pairs encoded as integers
  # input = 100 chars, output = 1 char
  for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

  return dataX, dataY


def prep_data(filename, seq_length):
  # create mapping of unique chars to integers
  raw_text = load_text(filename)
  n_chars = len(raw_text)

  chars = sorted(list(set(raw_text))) # all unique chars
  n_vocab = len(chars)
  char_to_int = dict((c, i) for i, c in enumerate(chars))

  dataX, dataY = inputs_and_outputs(raw_text, char_to_int, n_chars, seq_length)

  X, y = reshape(dataX, dataY, seq_length, n_vocab)

  return raw_text, chars, char_to_int, X, y
