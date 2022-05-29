import random
import tensorflow as tf
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

def play(r1, r2, replay_buffer):
    privs = [game.make_priv(r1, 0).to(device), game.make_priv(r2, 1).to(device)]

    def play_inner(state):
        cur = game.get_cur(state)
        calls = game.get_calls(state)
        assert cur == len(calls) % 2

        if calls and calls[-1] == game.LIE_ACTION:
            #prev call good if oppenent had to lie
            prev_call = calls[-2] if len(calls) >= 2 else -1
            res = 1 if game.evaluate_call(r1, r2, prev_call) else -1

        else:
            last_call = calls[-1] if calls else -1
            #currently set eps to 0 in policy
            eps = 0
            action = game.sample_action(privs[cur], state, last_call, eps)
            new_state = game.apply_action(state, action)
            # min/max
            res = -play_inner(new_state)

        # save info
        replay_buffer.append((privs[cur], state, res))
        replay_buffer.append((privs[1 - cur], state, -res))

        return res

    with tf.stop_gradient():
        state = game.make_state().to(device)
        play_inner(state)