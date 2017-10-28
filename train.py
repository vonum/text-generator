import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

from prep import prep_data

seq_length = 100
raw_text, chars, char_to_int, X, y = prep_data('alice-in-wonderland.txt', seq_length)

n_chars = len(raw_text)
n_vocab = len(chars)
print 'Total Characters: ', n_chars
print 'Total Vocab: ', n_vocab

n_patterns = len(X)
print 'Total Patterns: ', n_patterns

# define the LSTM model

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath='/output/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
