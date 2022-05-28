# this is the neural network

import random
import tensorflow as tf
import itertools
import numpy as np
import math
from collections import Counter


class Net(tf.keras.Model):

    def __init__(self, d_pri, d_pub):
        super().__init__()

        # want a bilinear layer that
        self.layers0 = tf.keras.layers.UpSampling2D(size=(d_pri, d_pub), interpolation='bilinear')
        # self.layer0 = tf.keras.layers.Dense(100, activation=tf.nn.relu)

        self.seq = tf.keras.models.Sequential([
            # First Dense layer
            tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)
        ])

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def forward(self, priv, pub):
        joined = self.layers0(priv, pub)
        return self.seq(joined)


class NetConcat(tf.keras.Model):

  def __init__(self, d_pri, d_pub):
    super().__init__()

    #output size of each layer: 500, 400, 300, 200, 100, 1(tanh)
    self.seq = tf.keras.models.Sequential([
      # First Dense layer input is d_priv + d_pub
      tf.keras.layers.Dense(units=500, activation=tf.nn.relu),
      tf.keras.layers.Dense(units=400, activation=tf.nn.relu),
      tf.keras.layers.Dense(units=300, activation=tf.nn.relu),
      tf.keras.layers.Dense(units=200, activation=tf.nn.relu),
      tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
      tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)
    ])

  def forward(self, priv, pub):
    if len(priv.shape) == 1:
      joined = tf.concat((priv, pub), axis=0)
    else:
      joined = tf.concat((priv, pub), axis=1)
    return self.seq(joined)

class NetCompBilin(tf.keras.Model):

  def __init__(self, d_pri, d_pub):
    super().__init__()

    middle = 500
    #input is size of d_private
    self.layer_pri = tf.keras.layers.Dense(units=middle)
    # input is size of d_pub
    self.layer_pub = tf.keras.layers.Dense(units=middle)

    self.seq = tf.keras.models.Sequential([
        # First Dense layer
        tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)
    ])

  def forward(self, priv, pub):
    joined = self.layer_pri(priv) * self.layer_pub(pub)
    return self.seq(joined)

class Resid(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        #input size is in_channels
        self.conv = tf.keras.layers.Conv1D(filters=out_channels, kernel_size=kernel_size)

    #check over if the formatting of y is correct
    def forward(self, x, y=None):
        y = self.conv(y)

class Net3(tf.keras.Model):
    def __init__(self, d_pri, d_pub):
        super.__init__()

        def conv(channels, size):
            model = tf.keras.models.Sequential([
                #input size = 1
                tf.keras.layers.Conv1D(channels, kernel_size=size),
                #input size is channels
                tf.keras.layers.BatchNormalization(),
                tf.keras.relu(),
                # input size = channels
                tf.keras.layers.Conv1D(channels, kernel_size=size),
                #input size is channels
                tf.keras.layers.BatchNormalization()
            ])
            return model

        self.layer0 = tf.keras.layers.UpSampling2D(size=(d_pri, d_pub), interpolation='bilinear')

        self.seq = tf.keras.models.Sequential([
            # First Dense layer
            tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)
        ])

    def forward(self, priv, pub):
        joined = self.layers0(priv, pub)
        return self.seq(joined)


class Net2(tf.keras.Model):
    def __init__(self, d_pri, d_pub):
        super.__init__()

        channels = 20
        self.left = tf.keras.models.Sequential([
            # input size = 1
            tf.keras.layers.Conv1D(channels, kernel_size=2),
            tf.keras.relu(),
            #input size = channels
            tf.keras.layers.Conv1D(1, kernel_size=2),
            tf.keras.relu()
        ])

        self.left = tf.keras.models.Sequential([
            # input size = 1
            tf.keras.layers.Conv1D(channels, kernel_size=2),
            tf.keras.relu(),
            # input size = channels
            tf.keras.layers.Conv1D(1, kernel_size=2),
            tf.keras.relu()
        ])

        # layers_size = 100
        self.bilin = tf.keras.layers.UpSampling2D(size=(d_pri, d_pub), interpolation='bilinear')

        self.seq = tf.keras.models.Sequential([
            # First Dense layer
            tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=100, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=1, activation=tf.nn.tanh)
        ])

    def forward(self, priv, pub):
        if len(priv.shape) == 1:
            assert len(pub.shape) == 1
            priv = tf.shape(tf.expand_dims(priv, 0))
            pub = tf.shape(tf.expand_dims(pub, 0))
        x = priv
        y = self.right(tf.shape(tf.expand_dims(pub, -2))).squeeze(1) + pub
        mixed = self.bilin(x, y)
        return self.seq(mixed)


def calc_args()

