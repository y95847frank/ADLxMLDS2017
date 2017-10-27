import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed, Reshape
from keras.layers import LSTM, Conv2D, MaxPooling2D, Convolution2D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
np.random.seed(7)
import LD
import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

TIME_STEP=777
INPUT_SIZE=39
BATCH_SIZE=16
NUM_CLASS=48
EPOCH=30

#x_test = LD.load_data('data', 'mfcc', 'test')
#np.save('data/mfcc/test.npy', x_test)
#x_train = LD.load_data('data', 'mfcc', 'train')
#np.save('data/mfcc/train.npy', x_train)
#y_train = LD.load_label('data', 'align_train.lab')
#np.save('data/label/label.npy', y_train)

if sys.argv[1] == 'fbank':
    INPUT_SIZE=69
    x_train = np.load('data/fbank/train.npy')
    x_test = np.load('data/fbank/test.npy')
else:
    x_train = np.load('data/mfcc/train.npy')
    x_test = np.load('data/mfcc/test.npy')
y_train = np.load('data/label/label_775.npy')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

img_rows, img_cols = TIME_STEP, INPUT_SIZE
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

x_val = x_train[:16]
y_val = y_train[:16]

model = Sequential()

model.add(Conv2D(24, (3, 1), input_shape=input_shape))
model.add(Conv2D(1, (1, 1), input_shape=model.output.shape))
#print(model.output.shape[1:-1])

model.add(Reshape((TIME_STEP-2, INPUT_SIZE)))

model.add(LSTM(64, return_sequences=True, batch_input_shape=(BATCH_SIZE, TIME_STEP-2, INPUT_SIZE)))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(Dense(48, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))

model.save_weights("./model/cnn24_l6_model.h5")
print("Saved model to disk")

# Final evaluation of the model
pred = model.predict(x_test, batch_size=16)
np.save('cnn24_l6.npy', pred)

