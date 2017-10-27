import numpy as np
import sys
import h5py
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
import LD

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

TIME_STEP=777
BATCH_SIZE=32
NUM_CLASS=48
EPOCH=60

#x_test = LD.load_data('data', 'mfcc', 'test')
#np.save('data/mfcc/test.npy', x_test)
#x_train = LD.load_data('data', 'mfcc', 'train')
#np.save('data/mfcc/train.npy', x_train)
#y_train = LD.load_label('data', 'align_train.lab')
#np.save('data/label/label.npy', y_train)

INPUT_SIZE = 69+39
mfcc_train = np.load('data/mfcc/train.npy')
fbank_train = np.load('data/fbank/train.npy')
x_train = np.concatenate((mfcc_train, fbank_train), axis=2)
y_train = np.load('data/label/m_label.npy')

print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Bidirectional(LSTM(512, recurrent_dropout=0.3, dropout=0.4, return_sequences=True), input_shape=(TIME_STEP, INPUT_SIZE)))
model.add(Dropout(0.3))
#model.add(Bidirectional(LSTM(512, recurrent_dropout=0.3, dropout=0.4, return_sequences=True)))
#model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(512, recurrent_dropout=0.3, dropout=0.4, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(512, recurrent_dropout=0.3, dropout=0.4, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dropout(0.3))
model.add(Dense(48, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
print(model.summary())
model_json = model.to_json()
with open("rlB.json", "w") as json_file:
    json_file.write(model_json)

checkpoint = ModelCheckpoint(filepath='rlB', monitor='val_loss', save_best_only=True, mode='min')
callbacks_list = [checkpoint]

hist = model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, validation_split=0.15, callbacks=callbacks_list)

# Final evaluation of the model

with open('rlB.hist', 'a') as fp:
    fp.writelines('{}:{}\n'.format(k,v) for k,v in hist.history.items())

