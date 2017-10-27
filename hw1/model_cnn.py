import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Reshape
from keras.layers import LSTM, Conv2D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
import LD
import sys


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

TIME_STEP=777
BATCH_SIZE=32
NUM_CLASS=48
EPOCH=40

#x_test = LD.load_data('data', 'mfcc', 'test')
#np.save('data/mfcc/test.npy', x_test)
#x_train = LD.load_data('data', 'mfcc', 'train')
#np.save('data/mfcc/train.npy', x_train)
#y_train = LD.load_label('data', 'align_train.lab')
#np.save('data/label/label.npy', y_train)

INPUT_SIZE=39+69
m_train = np.load('data/mfcc/train.npy')
f_train = np.load('data/fbank/train.npy')
x_train = np.concatenate((m_train, f_train), axis=2)

y_train = np.load('data/label/m_label.npy')

print(x_train.shape)
print(y_train.shape)

row, col = TIME_STEP, INPUT_SIZE
x_train = x_train.reshape(x_train.shape[0], row, col, 1)
input_shape = (row, col, 1)

model = Sequential()
model.add(Conv2D(64, (3, 1), input_shape=input_shape))
model.add(Conv2D(1, (1, 1), input_shape=model.output.shape))

model.add(Reshape((TIME_STEP-2, INPUT_SIZE)))

model.add(LSTM(64, return_sequences=True, batch_input_shape=(BATCH_SIZE, TIME_STEP, INPUT_SIZE)))
model.add(Dense(48, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model_json = model.to_json()
with open("cnn.json", "w") as json_file:
    json_file.write(model_json)

filepath='cnn.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, validation_split=0.15, callbacks=callbacks_list)

