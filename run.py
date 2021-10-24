import pickle
import math
import time
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import random
from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from sklearn.model_selection import train_test_split
from random import randint
from gensim.models import Word2Vec
from utils import rep, hit_ratio
from models import LinearVAE, LinearVAE2, return_models

print("Loading data.....")

### DATA NOT UPLOADED CURRENTLY .. WILL BE UPLOADED LATER ####
with open('./spotify_acoustic_11d.pkl', 'rb') as f:
    features_data = pickle.load(f)

'''
features_data is a dictionary with key as songs and values as 11 size spotify acoustic features in list format
Eg: {('Oscillate Wildly - 2011 Remaster', 'The Smiths'): [0.657,
  0.791,
  ...
  0.783,
  116.329],...}

'''

with open('./lyrical_features.p', 'rb') as f:
    lyrics = pickle.load(f) 

'''
lyrics is a dictionary with key as songs and values as 150 size sentence bert features in numpy array format
Eg: {('From Zero to Nothing',
  'Sybreed'): array([ 4.62941360e+00,  4.61279774e+00,  6.16572952e+00, -4.80036068e+00,
         ....
        -2.14760512e-01, -2.37200648e-01, -6.39073104e-02,  2.68031120e-01,
         5.13907708e-02, -4.18001950e-01], dtype=float32), ... }

'''

with open('./data_with_features_and_emotions_and_tags.p', 'rb') as f:
    data = pickle.load(f)


'''
data is a dictionary with key as users and values as their listening history list (Each history is a list of song ids)
Eg: {'user1': [[[1684, 13108, 3082, 23623]], [[35443, 1684, 15800, 13108, 3082]]], ...}
'''

with open('./tags_features.p', 'rb') as f:
    tags = pickle.load(f)

'''
tags is a dictionary with key as songs and values as 300 size tag features in list format
Eg: {('Scream', 'Adelitas Way'): [-0.07053999722003937,
  0.07077999740839004,
  -0.12173999786376953,
  ....
  0.07528000235557557,
  0.02007999837398529], ... }

'''

print("Preprocesssing Data...")

data2 = {}
sentences = []
for user in data:
    x = []
    for session in data[user]:
        req = [session[0]]
        x += req
        sentences += req
    data2[user] = x

arr = []
for history in sentences:
    arr += history
    
song_to_ind = {}
ind_to_song = {}
vocab_sz = 0
songs = set(arr)
for song in songs:
    song_to_ind[song] = vocab_sz
    ind_to_song[vocab_sz] = song
    vocab_sz += 1
extra_tokens = ['<pad>']

print("Song vocabulary size: ", vocab_sz)

for token in extra_tokens:
    song_to_ind[token] = vocab_sz
    ind_to_song[vocab_sz] = token
    vocab_sz += 1

for user in data2:
    for i in range(len(data2[user])):
        for j in range(len(data2[user][i])):
            data2[user][i][j] = song_to_ind[data2[user][i][j]]

sentences = []
for user in data2:
    for session in data2[user]: 
        req = []
        for song in session:
            req.append(str(song))
        sentences += [req]
wordvec = Word2Vec(sentences, vector_size=150, min_count=1) 

initial_lyrics_vector = np.zeros(150)
for song in lyrics:
    initial_lyrics_vector += lyrics[song]
initial_lyrics_vector /= len(lyrics)


autoENC = LinearVAE()
autoENC.load_state_dict(torch.load('./models/vae', map_location=torch.device('cpu')))
autoENC2 = LinearVAE2()
autoENC2.load_state_dict(torch.load('./models/vae2', map_location=torch.device('cpu')))

matrix_len = vocab_sz
weights_matrix = np.zeros((matrix_len, 150))
weights_matrix1 = np.zeros((matrix_len, 150))
weights_matrix2 = np.zeros((matrix_len, 150))
weights_matrix3 = np.zeros((matrix_len, 150))
for i in range(vocab_sz - 1):
    song = ind_to_song[i]
    inp = features_data[song]
    temp = rep(inp)
    temp = torch.tensor(temp)
    temp = autoENC(temp.float(), 1)[0].tolist()
    temp2 = wordvec.wv[str(i)].tolist()
    weights_matrix[i] = rep(temp2)
    weights_matrix1[i] = rep(temp)
    lyric_vals = []
    if song in lyrics:
        lyric_vals = lyrics[song].tolist()
    else:
        lyric_vals = initial_lyrics_vector
    weights_matrix2[i] = rep(lyric_vals)
    inp = tags[song]
    temp = rep(inp)
    temp = torch.tensor(temp)
    temp = autoENC2(temp.float(), 1)[0].tolist()
    weights_matrix3[i] = rep(temp)

weights_matrix = torch.tensor(weights_matrix)
weights_matrix1 = torch.tensor(weights_matrix1)
weights_matrix2 = torch.tensor(weights_matrix2)
weights_matrix3 = torch.tensor(weights_matrix3)

complete = []
for user in data2:
    for session in data2[user]:
        window = len(session)
        for i in range(0, len(session), window):
            req = session[i: i + window]
            if (len(req) >= 4):
                temp = []
                for ind in req:
                    temp.append((ind, features_data[ind_to_song[ind]]))
                complete += [temp]
temp_train_data, temp_test_data = train_test_split(complete, test_size=0.30, random_state=42)
print(len(temp_train_data), len(temp_test_data))
train_data = []
test_data = []
for session in temp_train_data:
    window = 4
    for i in range(0, len(session), window):
        req = session[i: i + window]
        if (len(req) >= 4):
            train_data += [req]
for session in temp_test_data:
    window = 4
    for i in range(0, len(session), window):
        req = session[i: i + window]
        if (len(req) >= 4):
            test_data += [req]
print("Training data size: ", len(train_data), "Testing data size: ", len(test_data))

print("Preparing Data loaders...")

BATCH_SIZE = 64
PAD_IDX = song_to_ind['<pad>']
def generate_batch(data_batch):
    batch = []
    target = []
    lengths = []
    indices = []
    for item in data_batch:
        total_len = len(item)
        to_app = []
        to_app_ind = []
        for i in range(total_len - 1):
            to_app.append(item[i][1])
            to_app_ind.append(item[i][0])
        batch.append(torch.tensor(to_app))
        indices.append(torch.tensor(to_app_ind))
        target.append(item[total_len - 1][0])
        lengths.append(total_len - 1)
    batch = pad_sequence(batch, padding_value=PAD_IDX)
    indices = pad_sequence(indices, padding_value=PAD_IDX)
    target = torch.tensor(target)
    lengths = torch.tensor(lengths)
    return batch, target, lengths, indices

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)
m = nn.LogSoftmax(dim=1)

Encoder, Attention, Model = return_models(weights_matrix, weights_matrix1, weights_matrix2, weights_matrix3)
enc = Encoder(vocab_sz, 150, 150, 150, 150, 0.2).to(device)
attn = Attention(150, 150, 150).to(device)
model = Model(enc, attn, 150, vocab_sz).to(device)

optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

print("Starting training...")

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for _, (src, trg, lens, inds) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        inds = inds.to(device)
        optimizer.zero_grad()
        output = model(src, lens, inds)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):
    model.eval()
    epoch_loss = 0
    hits = [0, 0, 0, 0, 0]
    total = [0, 0, 0, 0, 0]
    with torch.no_grad():
        for _, (src, trg, lens, inds) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            inds = inds.to(device)
            output = model(src, lens, inds)
            logit = m(output)
            arr = hit_ratio(logit, trg)
            loss = criterion(logit, trg)
            for i in range(len(hits)):
                hits[i] += arr[i]
                total[i] += src.shape[1]
            epoch_loss += loss.item()
    print("Hit ratio for k = 10: ", (hits[0] / total[0]) * 100)
    print("Hit ratio for k = 20: ", (hits[1] / total[1]) * 100)
    print("Hit ratio for k = 30: ", (hits[2] / total[2]) * 100)
    print("Hit ratio for k = 40: ", (hits[3] / total[3]) * 100)
    print("Hit ratio for k = 50: ", (hits[4] / total[4]) * 100)
    return epoch_loss / len(iterator)

def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    start_time = time.time()
    for i in range(3):
        train_loss = train(model, train_iter, optimizer, criterion)
    test_loss = evaluate(model, test_iter, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print('Epoch: '+str(epoch)+' | Time: '+str(epoch_mins)+'m '+str(epoch_secs)+'s')
    print('\tTrain Loss: '+str(train_loss)+' | Train PPL: '+str(math.exp(train_loss))+'')
    print('\tTest Loss: '+str(test_loss)+' | Test PPL: '+str(math.exp(test_loss))+'')