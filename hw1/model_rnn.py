import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
np.random.seed(7)
import LD
import sys

TIME_STEP=777
INPUT_SIZE=39
BATCH_SIZE=16
NUM_CLASS=48
EPOCH=20

#x_test = LD.load_data('data', 'mfcc', 'test')
#np.save('data/mfcc/test.npy', x_test)
#x_train = LD.load_data('data', 'mfcc', 'train')
#np.save('data/mfcc/train.npy', x_train)
#y_train = LD.load_label('data', 'align_train.lab')
#np.save('data/label/label.npy', y_train)

if sys.argv[1] == 'fbank':
    INPUT_SIZE = 69
    x_train = np.load('data/fbank/train.npy')
    x_test = np.load('data/fbank/test.npy')
else:
    x_train = np.load('data/mfcc/train.npy')
    x_test = np.load('data/mfcc/test.npy')
y_train = np.load('data/label/label.npy')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

x_val = x_train[:16]
y_val = y_train[:16]

model = Sequential()
model.add(LSTM(64, return_sequences=True, batch_input_shape=(BATCH_SIZE, TIME_STEP, INPUT_SIZE)))
model.add(LSTM(64, return_sequences=True, stateful=True))
model.add(LSTM(64, return_sequences=True, stateful=True))
model.add(LSTM(64, return_sequences=True, stateful=True))
model.add(LSTM(64, return_sequences=True, stateful=True))
model.add(LSTM(64, return_sequences=True, stateful=True))
model.add(Dense(48, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))

model_json = model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model2.h5")
print("Saved model to disk")

# Final evaluation of the model
pred = model.predict(x_test, batch_size=16)
np.save('second.npy', pred)

#scores = model.evaluate(x_val, y_val, batch_size=1)
#print("Accuracy: %.2f%%" % (scores[1]*100))
