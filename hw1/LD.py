import sys
import os
import numpy as np

MAX_LEN = 777
VEC_NUM = 39
CLASS_NUM = 48

def load_all_data(data, filename):
    VEC_NUM = 69+39
    file_1 = open(os.path.join(data, 'mfcc', filename+'.ark'), 'r')
    file_2 = open(os.path.join(data, 'fbank', filename+'.ark'), 'r')
    content1 = file_1.readlines()
    content2 = file_2.readlines()
    
    data = np.empty((0,MAX_LEN,VEC_NUM), float)
    last = ''
    tmp = []
    first = True
    f_batch = True
    for (c1, c2) in zip(content1, content2):
        ID = c1.split(' ')[0]
        vec1 = c1.split()[1:]
        vec2 = c2.split()[1:]
        vec1 = [float(v) for v in vec1]
        vec2 = [float(v) for v in vec2]
        vec = vec1+vec2
        index = ID.rfind('_')
        ID = ID[0:index]
        if first or ID == last :
            tmp.append(vec)
            last = ID
            first = False
        else:
            if len(tmp) < MAX_LEN:
                for i in range(MAX_LEN-len(tmp)):
                    tmp.append([0]*len(tmp[0]))
            tmp = np.asarray(tmp)
            tmp = tmp.reshape(1, tmp.shape[0], tmp.shape[1])
            data = np.append(data, tmp, axis=0)
            last = ID
            tmp = []
            tmp.append(vec)
    if len(tmp) < MAX_LEN:
        for i in range(MAX_LEN-len(tmp)):
            tmp.append([0]*len(tmp[0]))
    tmp = np.asarray(tmp)
    tmp = tmp.reshape(1, tmp.shape[0], tmp.shape[1])
    data = np.append(data, tmp, axis=0)
    return data

def load_data(data, mode, filename):
    if mode == 'fbank':
        VEC_NUM = 69
    else:
        VEC_NUM = 39
    file_d = open(os.path.join(data, mode, filename+'.ark'), 'r')
    content = file_d.readlines()
    
    data = np.empty((0,MAX_LEN,VEC_NUM), float)
    last = ''
    tmp = []
    first = True
    f_batch = True
    for c in content:
        ID = c.split(' ')[0]
        vec = c.split()[1:]
        vec = [float(v) for v in vec]
        index = ID.rfind('_')
        ID = ID[0:index]
        if first or ID == last :
            tmp.append(vec)
            last = ID
            first = False
        else:
            if len(tmp) < MAX_LEN:
                for i in range(MAX_LEN-len(tmp)):
                    tmp.append([0]*len(tmp[0]))
            tmp = np.asarray(tmp)
            #tmp = (tmp - np.mean(tmp, axis=0)) / np.std(tmp, axis=0)
            tmp = tmp.reshape(1, tmp.shape[0], tmp.shape[1])
            data = np.append(data, tmp, axis=0)
            last = ID
            tmp = []
            tmp.append(vec)
    if len(tmp) < MAX_LEN:
        for i in range(MAX_LEN-len(tmp)):
            tmp.append([0]*len(tmp[0]))
    tmp = np.asarray(tmp)
    #tmp = (tmp - np.mean(tmp, axis=0)) / np.std(tmp, axis=0)
    tmp = tmp.reshape(1, tmp.shape[0], tmp.shape[1])
    data = np.append(data, tmp, axis=0)
    return data

def load_label(data, filename):
    file_l = open(os.path.join(data, "label", filename), 'r')
    file_phone = open(os.path.join(data, "48phone_char.map"), 'r')
    mapping = open(os.path.join(data, 'phones', '48_39.map'), 'r')
    p_map = dict()
    mappings = mapping.readlines()
    for m in mappings:
        m = m.split()
        p_map[m[0]] = m[1]

    phone = dict()
    lines = file_phone.readlines()
    for line in lines:
        line = line.split('\t')
        phone[line[0]] = int(line[1])

    content = file_l.readlines()
    label = np.empty((0,MAX_LEN, CLASS_NUM), int)
    last = ''
    tmp = []
    first = True
    for c in content:
        ID, Label = c.split(',')
        index = ID.rfind('_')
        ID = ID[0:index]
        Label = Label.replace('\n', '')
        if first or ID == last :
            p = [0]*CLASS_NUM
            p[phone[p_map[Label]]] = 1
            tmp.append(p)
            last = ID
            first = False
        else:
            if len(tmp) < MAX_LEN:
                p = [0]*CLASS_NUM
                p[37] = 1
                length = MAX_LEN-len(tmp)
                for i in range(length):
                    tmp.append(p)
            count =  len(tmp) - MAX_LEN
            for i in range(count):
                del tmp[-1]
            tmp = np.asarray(tmp)
            tmp = tmp.reshape((1, tmp.shape[0], tmp.shape[1]))
            label = np.append(label, tmp, axis=0)
            last = ID
            tmp = []
            p = [0]*CLASS_NUM
            p[phone[p_map[Label]]] = 1
            tmp.append(p)

    if len(tmp) < MAX_LEN:
        p = [0]*CLASS_NUM
        p[37] = 1
        length = MAX_LEN-len(tmp)
        for i in range(length):
            tmp.append(p)
    tmp = np.asarray(tmp)
    tmp = tmp.reshape((1, tmp.shape[0], tmp.shape[1]))
    label = np.append(label, tmp, axis=0)

    return label
