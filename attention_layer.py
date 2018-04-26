# -*- coding: utf-8 -*-
# file: attention_layer.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer

class Attention(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.ATTENTION_SIZE = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.EMBED_SIZE = input_shape[0][-1]
        if self.ATTENTION_SIZE is None:
            self.ATTENTION_SIZE = self.EMBED_SIZE
        # W: (EMBED_SIZE, ATTENTION_SIZE,)
        # b: (ATTENTION_SIZE,)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(self.EMBED_SIZE, self.ATTENTION_SIZE,),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(self.ATTENTION_SIZE,),
                                 initializer="zeros",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs, mask=None):
        x, u = inputs
        if u is None:
            u = self.add_weight(name="u_{:s}".format(self.name),
                                     shape=(self.ATTENTION_SIZE,),
                                     initializer="glorot_normal",
                                     trainable=True)
        # u: (?, ATTENTION_SIZE,)
        # x: (?, MAX_TIMESTEPS, EMBED_SIZE)
        # ut: (?, MAX_TIMESTEPS, ATTENTION_SIZE)
        ut = K.tanh(K.dot(x, self.W) + self.b)
        # at: (?, MAX_TIMESTEPS,)
        at = K.batch_dot(ut, u)
        at = K.softmax(at)
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (?, MAX_TIMESTEPS, EMBED_SIZE,)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (?, EMBED_SIZE,)
        output = K.sum(ot, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])