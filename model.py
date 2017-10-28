from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

def create_lstm_model(X_time_stamps, X_features, y_dim):
  model = Sequential()
  model.add(LSTM(256, input_shape=(X_time_stamps, X_features), return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(256))
  model.add(Dropout(0.2))
  model.add(Dense(y_dim, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')

  return model
