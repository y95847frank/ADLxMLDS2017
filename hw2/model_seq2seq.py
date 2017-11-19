# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import numpy as np
import string
import re
import random
import json
import csv
import sys
import os
import time
import math
from bleu_eval import BLEU
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 80

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

class Label:
    def __init__(self, name, count):
        self.name = name
        self.count = count
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        elif self.word2count[word] == self.count:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] += 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

'''
def get_best(captions,caption_num=1):
    captions = list(map(str,captions))
    score_list = [[best(captions[y_index],captions[:y_index]+captions[y_index+1:]),y_index] for y_index in range(len(captions))]
    score_list = sorted(score_list, key = lambda x : x[0], reverse=True)
    selected_list = [captions[score_list[k][1]] for k in range(caption_num)]
    return selected_list
'''

def get_data(video_label_path, video_feat_path, word_count=2):

    captions = []
    pairs = []
    fp = open(video_label_path, 'r')
    label = json.load(fp)
    for d in label:
        tmp = [normalizeString(x.replace('.', '')) for x in d['caption']]
        captions += [normalizeString(x.replace('.', '')) for x in d['caption']]
        pairs.append([np.load(video_feat_path+d['id']+'.npy'), tmp])
    print("Read %s sentence" % len(captions))
    output_lang = Label('train', word_count)
    for cap in captions:
        output_lang.addSentence(cap)
    print("Counted words:")
    print(output_lang.name, output_lang.n_words)

    return output_lang, captions, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ') if word in lang.word2index]

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def feat2variable(feat):
    result = Variable(torch.FloatTensor(feat))
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(output_lang, pair):
    input_variable = feat2variable(pair[0])
    target_variable = variableFromSentence(output_lang, random.choice(pair[1]))
    return (input_variable, target_variable)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        #embedded = self.embedding(input).view(1, 1, -1)
        embedded = input.view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
   
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    decoder_hidden = encoder_hidden
    #decoder_hidden = decoder.initHidden()

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            
            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def trainIters(encoder, decoder, n_iters, print_every=1300, plot_every=100, learning_rate=0.001, epoch=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    t_pair = pairs[:1300]
    x_pair = pairs[1300:]
    best_score = 0
    count_down = 0
    timer = 0
    for e in range(epoch):
        training_pairs = [variablesFromPair(output_lang, t_pair[i]) for i in range(n_iters)]
        criterion = nn.NLLLoss()
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_variable = training_pair[0]
            target_variable = training_pair[1]
     
            loss = train(input_variable, target_variable, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('epoch : %d %s (%d %d%%) %.4f' % (e, timeSince(start, iter*(e+1) / n_iters/epoch),
                                             iter, (e+1)*iter / n_iters / epoch * 100, print_loss_avg))
        score = test(encoder, decoder, x_pair)
        if score > best_score:
            torch.save(encoder.state_dict(), './model/encoder_exp_best.pkl')
            torch.save(decoder.state_dict(), './model/decoder_exp_best.pkl')
            best_score = score
            timer = 0
        else:
            timer += 1
        print('score = '+str(score)+" best_score = "+str(best_score))
        if timer > 12:
            break

def test(encoder, decoder, pairs):
    #evaluateRandomly(encoder, decoder)
    score = 0.0
    for i in range(len(pairs)):
        output, att = evaluate(encoder, decoder, pairs[i][0])
        out = ' '.join(output[:-1])
        tmp = 0.0
        score += BLEU(out, pairs[i][1], True)
        '''
        for ans in pairs[i][1]:
            tmp += BLEU(out, ans)
        score += (tmp/len(pairs[i][1]))
        '''

    return (score / len(pairs))

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = feat2variable(sentence)
    #input_variable = variableFromSentence(output_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=4):
    for i in range(n):
        pair = pairs[i]
        #print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

teacher_forcing_ratio = 0.8

print("reading data!")
output_lang, captions, pairs = get_data('./MLDS_hw2_data/training_label.json', './MLDS_hw2_data/training_data/feat/')
#print('dump output_lang into lang/lang_'+sys.argv[1]+'.obj')
#pickle.dump(output_lang, open("lang/lang_"+sys.argv[1]+".obj", "wb"))
#print(random.choice(pairs)[1])

hidden_size = 256
encoder1 = EncoderRNN(4096, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                               1, dropout_p=0.1)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

print("Start training!")
trainIters(encoder1, attn_decoder1, 1300, epoch=40, print_every=1300)

#print("Start evaluate!")
#evaluateRandomly(encoder1, attn_decoder1)


