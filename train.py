import random
import tensorflow as tf
from torch import nn
import itertools
import math
from collections import Counter
import argparse
import os

from nnetwork import *

#Number of dice for player 1
d1 = 2
#Number of dice for player 2
d2 = 2
#Number of sides on the dice
sides = 6
#one of normal, joker, stairs
type = "normal"

D_PUB, D_PRI, *_ = calc_args(d1, d2, sides, type)
model = NetConcat(D_PRI, D_PUB)
game = Game(d1, d2, sides, type)

device = tf.device("cuda")
model.to(device)

tf.stop_gradient()

