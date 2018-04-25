# -*- coding: utf-8 -*-
# file: utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

def load_word_vec(word_index=None, embedding_dim=100):
    fname = './glove.twitter.27B/glove.twitter.27B.'+str(embedding_dim)+'d.txt'
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if tokens[0] in word_index.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def read_twitter(type='train', embedding_dim=100, max_seq_len=40, max_aspect_len=3):
    print("preparing twitter data...")
    fname = './datasets/acl-14-short-data/'+type+'.raw'
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    print("number of twitters:", len(lines)/3)

    text = ''
    texts_raw = []
    texts_left = []
    texts_right = []
    aspects = []
    polarities = []

    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i+1].lower().strip()
        polarity = lines[i+2].strip()

        text_raw = text_left + " " + aspect + " " + text_right
        text += text_raw

        texts_raw.append(text_raw)
        texts_left.append(text_left)
        texts_right.append(text_right)
        aspects.append(aspect)
        polarities.append(int(polarity))

    text_words = text.strip().split()
    print('tokenizing...')
    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(text_words)
    word_index = tokenizer.word_index

    texts_raw_indices = tokenizer.texts_to_sequences(texts_raw)
    texts_raw_indices = pad_sequences(texts_raw_indices, maxlen=max_seq_len*2+max_aspect_len)
    texts_left_indices = tokenizer.texts_to_sequences(texts_left)
    texts_left_indices = pad_sequences(texts_left_indices, maxlen=max_seq_len)
    texts_right_indices = tokenizer.texts_to_sequences(texts_right)
    texts_right_indices = pad_sequences(texts_right_indices, maxlen=max_seq_len, padding='post', truncating='post') # 方向不同
    aspects_indices = tokenizer.texts_to_sequences(aspects)
    aspects_indices = pad_sequences(aspects_indices, maxlen=max_aspect_len)
    polarities_matrix = K.eval(tf.one_hot(indices=polarities, depth=3))

    if type == 'test':
        return texts_raw_indices, texts_left_indices, aspects_indices, texts_right_indices, polarities_matrix

    embedding_matrix_file_name = str(embedding_dim)+'_twitter_embedding_matrix.dat'
    if os.path.exists(embedding_matrix_file_name):
        print('loading twitter embedding_matrix...')
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors...')
        word_vec = load_word_vec(word_index, embedding_dim)
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        print('building twitter embedding_matrix...')
        for word, i in word_index.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))

    return texts_raw_indices, texts_left_indices, aspects_indices, texts_right_indices, polarities_matrix, \
           embedding_matrix, \
           tokenizer

if __name__ == '__main__':
    read_twitter()
