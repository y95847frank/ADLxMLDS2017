# MLP for Pima Indians Dataset serialize to YAML and HDF5
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import numpy as np
import os
import sys
from itertools import groupby
import LD

m_test = LD.load_data(sys.argv[4], 'mfcc', 'test')
f_test = LD.load_data(sys.argv[4], 'fbank', 'test')

x_test = np.concatenate((m_test, f_test), axis=2)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
#print x_test.shape
# load weights into new model
json_file = open(sys.argv[1], 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(sys.argv[2])

#print(loaded_model.summary())

result = loaded_model.predict(x_test, batch_size=1)
#np.save(sys.argv[3], pred)

file_phone = open(os.path.join(sys.argv[4], "48phone_char.map"), 'r')
phone = dict()
lines = file_phone.readlines()
for line in lines:
    line = line.split('\t')
    phone[int(line[1])] = line[2][:-1]
phone[3] = phone[0]
phone[5] = phone[2]
phone[9] = phone[37]
phone[14] = phone[27]
phone[15] = phone[29]
phone[16] = phone[37]
phone[23] = phone[22]
phone[43] = phone[37]
phone[47] = phone[36]

ans = []
for line in result:
    tmp = []
    for w in line:
        tmp.append(phone[np.argmax(w)])
    ans.append(tmp)

#np.save(sys.argv[2]+'.npy', ans)

for i, line in enumerate(ans):
    group = ans[i]
    '''
    group = []
    for j in range(2, len(ans[i])-2):
        if ans[i][j-2]==ans[i][j] or ans[i][j-1]==ans[i][j] or ans[i][j+1]==ans[i][j] or ans[i][j+2]==ans[i][j]:
            group += ans[i][j]
    group = ans[i][:2] + group + ans[i][-2:]
    '''
    tmp = []
    for x,y in groupby(group):
        k = len(list(y))
        if k > 1:
            for m in range(k):
                tmp.append(x)
    group = tmp
    
    group = [x for x,y in groupby(group) if len(list(y)) > 2]
    group = [x[0] for x in groupby(group)]
    if group[0] == 'L':
        del group[0]
    if group[-1] == 'L':
        del group[-1]
    ans[i] = group

fp = open('sample.csv', 'r')
fw = open(sys.argv[3], 'w')
fw.write("id,phone_sequence\n")

content = fp.readlines()
for (c, line) in zip(content[1:], ans):
    fw.write(c[:-1])
    for w in line:
        fw.write(w[0])
    fw.write('\n')


