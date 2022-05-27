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


class Net(tf.keras.Model):

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