import sys
import numpy as np
import os
import skimage
import skimage.io
import skimage.transform

def gen_test(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    all_hair = ['orange', 'white', 'aqua', 'gray','green', 'red', 'purple', 'pink','blue', 'black', 'brown', 'blonde']
    all_eyes = ['gray', 'black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 'brown', 'red', 'blue']
    vectors = []
    names = []

    for l in lines:
        ID = l.split(',')[0]
        tags = l.strip().split(',')[1].split()
        hair_hot = np.zeros(len(all_hair))
        eyes_hot = np.zeros(len(all_eyes))
        if tags[1] == 'hair':
            hair_hot[all_hair.index(tags[0])] = 1
            if len(tags) > 2:
                eyes_hot[all_eyes.index(tags[2])] = 1
        else:
            eyes_hot[all_eyes.index(tags[0])] = 1
            if len(tags) > 2:
                hair_hot[all_hair.index(tags[2])] = 1
        vec = np.concatenate((hair_hot, eyes_hot))
        vectors.append(vec)
        names.append(ID)
    return np.asarray(names), np.asarray(vectors)

def load_data(path):
    f = open(os.path.join(path, 'tags_clean.csv'), 'r')
    lines = f.readlines()
    f.close()
    all_hair = ['orange', 'white', 'aqua', 'gray','green', 'red', 'purple', 'pink','blue', 'black', 'brown', 'blonde']
    all_eyes = ['gray', 'black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 'brown', 'red', 'blue']

    vectors = []
    images = []

    for l in lines:
        ID = l.split(',')[0]
        tags = l.strip().split(',')[1].split('\t')
        tags = [t.split(':')[0] for t in tags]
        
        hair_color = ''
        eyes_color = ''
        for t in tags:
            try:
                if t.split()[1] == 'hair' and t.split()[0] in all_hair:
                    hair_color = t.split()[0]
                elif t.split()[1] == 'eyes' and t.split()[0] in all_eyes:
                    eyes_color = t.split()[0]
            except:
                pass
        if hair_color != '' and eyes_color != '':
            hair_hot = np.zeros(len(all_hair))
            hair_hot[all_hair.index(hair_color)] = 1
            eyes_hot = np.zeros(len(all_eyes))
            eyes_hot[all_eyes.index(eyes_color)] = 1

            vec = np.concatenate((hair_hot, eyes_hot))

            i = skimage.io.imread(os.path.join(path, 'faces', ID+'.jpg'))
            i = skimage.transform.resize(i, (64, 64))
            i = np.array(i)

            vectors.append(vec)
            images.append(i)
    np.save(os.path.join(path, 'text.npy'), vectors)
    np.save(os.path.join(path, 'img.npy'), images)
    return vectors, images
                    
