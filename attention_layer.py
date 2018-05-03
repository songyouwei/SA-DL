# -*- coding: utf-8 -*-
# file: attention_layer.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer

class Attention(Layer):
    def __init__(self, score_function='scaled_dot_product', initializer='glorot_normal', regularizer=None, **kwargs):
        # score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        self.score_function = score_function
        self.initializer = initializer
        self.regularizer = regularizer
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.EMBED_DIM = input_shape[0][-1].value
        K_LEN = input_shape[0][1].value
        Q_LEN = input_shape[1][1].value if input_shape[1].ndims == 3 else 1

        if self.score_function == 'mlp':
            self.W1 = self.add_weight(name="W1_{:s}".format(self.name),
                                        shape=(Q_LEN, Q_LEN + K_LEN,),
                                        initializer=self.initializer,
                                        regularizer=self.regularizer,
                                        trainable=True)
            self.W2 = self.add_weight(name="W2_{:s}".format(self.name),
                                        shape=(self.EMBED_DIM, K_LEN,),
                                        initializer=self.initializer,
                                        regularizer=self.regularizer,
                                        trainable=True)
        elif self.score_function == 'bi_linear':
            self.W = self.add_weight(name="W_{:s}".format(self.name),
                                        shape=(self.EMBED_DIM, self.EMBED_DIM,),
                                        initializer=self.initializer,
                                        regularizer=self.regularizer,
                                        trainable=True)

        super(Attention, self).build(input_shape)

    def call(self, inputs, mask=None):
        # output = softmax(score)
        k, q = inputs
        if len(q.shape) == 2:
            q = K.expand_dims(q, axis=1)
        # k: (?, K_LEN, EMBED_DIM,)
        # q: (?, Q_LEN, EMBED_DIM,)
        # score: (?, Q_LEN, K_LEN,)
        if self.score_function == 'scared_dot_product':
            kt = K.permute_dimensions(k, (0, 2, 1))
            qkt = K.batch_dot(q, kt)
            score = qkt / self.EMBED_DIM
        elif self.score_function == 'mlp':
            kq = K.concatenate([k, q], axis=1)
            kqw2 = K.tanh(K.dot(kq, self.W2))
            score = K.permute_dimensions(K.dot(self.W1, kqw2), (1, 0, 2))
        elif self.score_function == 'bi_linear':
            qw = K.dot(q, self.W)
            kt = K.permute_dimensions(k, (0, 2, 1))
            score = K.batch_dot(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = K.softmax(score)
        if mask is not None:
            score *= K.cast(mask, K.floatx())
        # output: (?, Q_LEN, EMBED_DIM,)
        output = K.batch_dot(score, k)
        return output

    def compute_output_shape(self, input_shape):
        # (?, Q_LEN, EMBED_DIM,)
        return (input_shape[0][0], input_shape[1][1], input_shape[0][-1])