# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from __future__ import print_function
import argparse
import os
from tensorflow.python.keras.callbacks import TensorBoard, LambdaCallback
# from tensorflow.python.keras.utils import plot_model
from utils import read_dataset
from custom_metrics import f1
from attention_layer import Attention
import tensorflow as tf
from tensorflow.python.keras import initializers, regularizers, optimizers, backend as K
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, Dense, Activation, LSTM, Embedding, Bidirectional, Lambda, Flatten


class RecurrentAttentionMemory:
    def __init__(self):
        self.HOPS = 5
        self.SCORE_FUNCTION = 'mlp'  # scaled_dot_product / mlp (concat) / bi_linear (general dot)
        self.DATASET = 'restaurant'  # 'twitter', 'restaurant', 'laptop'
        self.POLARITIES_DIM = 3
        self.EMBEDDING_DIM = 300
        self.LEARNING_RATE = 0.001
        self.INITIALIZER = initializers.RandomUniform(minval=-0.05, maxval=0.05)
        self.REGULARIZER = regularizers.l2(0.001)
        self.LSTM_PARAMS = {
            'units': 200,
            'activation': 'tanh',
            'recurrent_activation': 'sigmoid',
            'kernel_initializer': self.INITIALIZER,
            'recurrent_initializer': self.INITIALIZER,
            'bias_initializer': self.INITIALIZER,
            'kernel_regularizer': self.REGULARIZER,
            'recurrent_regularizer': self.REGULARIZER,
            'bias_regularizer': self.REGULARIZER,
            'dropout': 0,
            'recurrent_dropout': 0,
        }
        self.MAX_SEQUENCE_LENGTH = 80
        self.MAX_ASPECT_LENGTH = 10
        self.BATCH_SIZE = 200
        self.EPOCHS = 100

        self.texts_raw_indices, self.texts_raw_without_aspects_indices, self.texts_left_indices, self.texts_left_with_aspects_indices, \
        self.aspects_indices, self.texts_right_indices, self.texts_right_with_aspects_indices, \
        self.polarities_matrix, \
        self.embedding_matrix, \
        self.tokenizer = \
            read_dataset(type=self.DATASET,
                         mode='train',
                         embedding_dim=self.EMBEDDING_DIM,
                         max_seq_len=self.MAX_SEQUENCE_LENGTH, max_aspect_len=self.MAX_ASPECT_LENGTH)

        if os.path.exists('ram_saved_model.h5'):
            print('loading saved model...')
            self.model = load_model('ram_saved_model.h5')
        else:
            print('Build model...')
            inputs_sentence = Input(shape=(self.MAX_SEQUENCE_LENGTH,), name='inputs_sentence')
            inputs_aspect = Input(shape=(self.MAX_ASPECT_LENGTH,), name='inputs_aspect')
            nonzero_count = Lambda(lambda xin: tf.count_nonzero(xin, dtype=tf.float32))(inputs_aspect)
            sentence = Embedding(input_dim=len(self.tokenizer.word_index) + 1,
                          output_dim=self.EMBEDDING_DIM,
                          input_length=self.MAX_SEQUENCE_LENGTH,
                          mask_zero=True,
                          weights=[self.embedding_matrix],
                          trainable=False, name='sentence_embedding')(inputs_sentence)
            aspect = Embedding(input_dim=len(self.tokenizer.word_index) + 1,
                             output_dim=self.EMBEDDING_DIM,
                             input_length=self.MAX_ASPECT_LENGTH,
                             mask_zero=True,
                             weights=[self.embedding_matrix],
                             trainable=False, name='aspect_embedding')(inputs_aspect)
            memory = Bidirectional(LSTM(return_sequences=True, **self.LSTM_PARAMS), name='memory')(sentence)
            aspect = Bidirectional(LSTM(return_sequences=True, **self.LSTM_PARAMS), name='aspect')(aspect)
            x = Lambda(lambda xin: K.sum(xin[0], axis=1) / xin[1], name='aspect_mean')([aspect, nonzero_count])
            shared_attention = Attention(score_function=self.SCORE_FUNCTION,
                                         initializer=self.INITIALIZER, regularizer=self.REGULARIZER,
                                         name='shared_attention')
            for i in range(self.HOPS):
                x = shared_attention((memory, x))
            x = Flatten()(x)
            x = Dense(self.POLARITIES_DIM)(x)
            predictions = Activation('softmax')(x)
            model = Model(inputs=[inputs_sentence, inputs_aspect], outputs=predictions)
            model.summary()
            model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=self.LEARNING_RATE), metrics=['acc', f1])
            # plot_model(model, to_file='model.png')
            self.model = model

    def train(self):
        tbCallBack = TensorBoard(log_dir='./ram_logs', histogram_freq=0, write_graph=True, write_images=True)
        def modelSave(epoch, logs):
            if (epoch + 1) % 5 == 0:
                self.model.save('ram_saved_model.h5')
        msCallBack = LambdaCallback(on_epoch_end=modelSave)

        texts_raw_indices, texts_raw_without_aspects_indices, texts_left_indices, texts_left_with_aspects_indices, \
        aspects_indices, texts_right_indices, texts_right_with_aspects_indices, \
        polarities_matrix = \
            read_dataset(type=self.DATASET,
                         mode='test',
                         embedding_dim=self.EMBEDDING_DIM,
                         max_seq_len=self.MAX_SEQUENCE_LENGTH, max_aspect_len=self.MAX_ASPECT_LENGTH)

        self.model.fit([self.texts_raw_indices, self.aspects_indices], self.polarities_matrix,
                       validation_data=([texts_raw_indices, aspects_indices], polarities_matrix),
                       epochs=self.EPOCHS, batch_size=self.BATCH_SIZE, callbacks=[tbCallBack])


if __name__ == '__main__':
    model = RecurrentAttentionMemory()
    model.train()
