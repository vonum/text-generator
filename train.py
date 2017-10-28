import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

from prep import prep_data
from model import create_lstm_model

seq_length = 100
raw_text, chars, char_to_int, X, y = prep_data('alice-in-wonderland.txt', seq_length)

n_chars = len(raw_text)
n_vocab = len(chars)
print 'Total Characters: ', n_chars
print 'Total Vocab: ', n_vocab

n_patterns = len(X)
print 'Total Patterns: ', n_patterns

print X.shape
print y.shape

model = create_lstm_model(X.shape[1], X.shape[2], y.shape[1])

# define the checkpoint
filepath='/output/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
