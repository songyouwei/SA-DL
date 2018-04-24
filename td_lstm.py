# -*- coding: utf-8 -*-
# file: td_lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


from __future__ import print_function
import argparse
import os
import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard
# from tensorflow.python.keras.utils import plot_model
from utils import read_twitter
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, Dense, Activation, LSTM, Embedding, Concatenate


class TDLSTM:
    def __init__(self):
        self.POLARITIES_DIM = 3
        self.EMBEDDING_DIM = 100
        self.MAX_SEQUENCE_LENGTH = 40
        self.ITERATION = 500
        self.BATCH_SIZE = 200

        self.texts_raw_indices, self.texts_left_indices, self.aspects_indices, self.texts_right_indices, \
        self.polarities_matrix, \
        self.embedding_matrix, \
        self.tokenizer = \
            read_twitter(maxlen=self.MAX_SEQUENCE_LENGTH)

        self.left_input = np.concatenate((self.texts_left_indices, self.aspects_indices), axis=1)
        self.right_input = np.concatenate((self.texts_right_indices, self.aspects_indices), axis=1)

        if os.path.exists('td_lstm_saved_model.h5'):
            print('loading saved model...')
            self.model = load_model('td_lstm_saved_model.h5')
        else:
            print('Build model...')
            inputs_l = Input(shape=(self.MAX_SEQUENCE_LENGTH + 2,))
            inputs_r = Input(shape=(self.MAX_SEQUENCE_LENGTH + 2,))
            Embedding_Layer = Embedding(input_dim=len(self.tokenizer.word_index) + 1,
                          output_dim=self.EMBEDDING_DIM,
                          input_length=self.MAX_SEQUENCE_LENGTH*2,
                          weights=[self.embedding_matrix],
                          trainable=False)
            x_l = Embedding_Layer(inputs_l)
            x_r = Embedding_Layer(inputs_r)
            x_l = LSTM(self.EMBEDDING_DIM*2, dropout=0.2, recurrent_dropout=0.1)(x_l)
            x_r = LSTM(self.EMBEDDING_DIM*2, dropout=0.2, recurrent_dropout=0.1, go_backwards=True)(x_r)
            x = Concatenate()([x_l, x_r])
            x = Dense(self.POLARITIES_DIM)(x)
            predictions = Activation('softmax')(x)
            model = Model(inputs=[inputs_l, inputs_r], outputs=predictions)
            model.summary()
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
            # plot_model(model, to_file='model.png')
            self.model = model

    def train(self):
        tbCallBack = TensorBoard(log_dir='./td_lstm_logs', histogram_freq=0, write_graph=True, write_images=True)

        texts_raw_indices, texts_left_indices, aspects_indices, texts_right_indices, polarities_matrix = \
            read_twitter(type='test', maxlen=self.MAX_SEQUENCE_LENGTH)

        left_input = np.concatenate((texts_left_indices, aspects_indices), axis=1)
        right_input = np.concatenate((texts_right_indices, aspects_indices), axis=1)

        for i in range(1, self.ITERATION):
            print()
            print('-' * 50)
            print('Iteration', i)
            self.model.fit([self.left_input, self.right_input], self.polarities_matrix,
                           validation_data=([left_input, right_input], polarities_matrix),
                           batch_size=self.BATCH_SIZE, callbacks=[tbCallBack])
            if i % 5 == 0:
                self.model.save('td_lstm_saved_model.h5')
                print('model saved')


    def predict(self, sentence):
        texts_raw_indices = self.tokenizer.texts_to_sequences([sentence])
        texts_raw_indices = pad_sequences(texts_raw_indices, maxlen=self.MAX_SEQUENCE_LENGTH * 2)
        pred = self.model.predict(texts_raw_indices, verbose=0)[0]
        print('pred:', pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sentence', type=str, default=None, help='predict with sentence')
    args = parser.parse_args()
    model = TDLSTM()
    if args.sentence:
        model.predict(args.sentence)
    else:
        model.train()
