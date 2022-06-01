import random
import tensorflow as tf
from nnetwork import *

import numpy as np
#Number of dice for player 1
d1 = 2
#Number of dice for player 2
d2 = 2
#Number of sides on the dice
sides = 6
#one of normal, joker, stairs
type = "normal"
#weight decay
weightDecay = 1e-2
#
path = "here"
D_PUB, D_PRI, *_ = calc_args(d1, d2, sides, type)
model = NetConcat(D_PRI, D_PUB)
game = Game(model, d1, d2, sides, type)

device = tf.device("cuda")
#model.compile(device)

#tf.no_gradient()

def play(r1, r2, replay_buffer):
    privs = [game.make_priv(r1, 0), game.make_priv(r2, 1)]

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
            print(state)
            action = game.sample_action(privs[cur], state, last_call, eps)
            new_state = game.apply_action(state, action)
            # min/max
            res = -play_inner(new_state)

        # save info
        replay_buffer.append((privs[cur], state, res))
        replay_buffer.append((privs[1 - cur], state, -res))

        return res

    #with tf.stop_gradient():
    print(game.make_state())
    #state = game.make_state().to(device)
    state = game.make_state()
    return play_inner(state)


def print_strategy(state):
    total_v = 0
    total_cnt = 0
    for r1, cnt in sorted(Counter(game.rolls(0)).items()):
        priv = game.make_priv(r1, 0).to(device)
        v = model(priv, state)
        rs = tf.tensor(game.make_regrets(priv, state, last_call=-1))
        if rs.sum() != 0:
            rs /= rs.sum()
        strat = []
        for action, prob in enumerate(rs):
            n, d = divmod(action, game.SIDES)
            n, d = n + 1, d + 1
            if d == 1:
                strat.append(f"{n}:")
            strat.append(f"{prob:.2f}")
        print(r1, f"{float(v):.4f}".rjust(7), f"({cnt})", " ".join(strat))
        total_v += v
        total_cnt += cnt
    print(f"Mean value: {total_v / total_cnt}")


class ReciLR(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, optimizer, gamma=1, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(ReciLR, self).__init__(optimizer)

    def get_lr(self):
        return [
            base_lr / (self.last_epoch + 1) ** self.gamma
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]

    def _get_closed_form_lr(self):
        return [
            base_lr / (self.last_epoch + 1) ** self.gamma for base_lr in self.base_lrs
        ]


def train():
    optimizer = tf.keras.optimizers.Adam()
    scheduler = ReciLR(optimizer, gamma=0.5)
    value_loss = tf.keras.losses.MeanSquaredError()
    all_rolls = list(itertools.product(game.rolls(0), game.rolls(1)))
    for t in range(100_000):
        replay_buffer = []

        BS = 100  # Number of rolls to include
        for r1, r2 in (
            all_rolls if len(all_rolls) <= BS else random.sample(all_rolls, BS)
        ):
            replay_buffer = play(r1, r2, replay_buffer)
        print(replay_buffer)
        random.shuffle(replay_buffer)
        privs, states, y = zip(*replay_buffer)

        privs = tf.concat(privs, axis=0).to(device)
        states = tf.concat(states, axis=0).to(device)
        y = tf.tensor(y, dtype=tf.float).reshape(-1, 1).to(device)

        y_pred = model(privs, states)

        # Compute and print loss
        loss = value_loss(y_pred, y)
        print(t, loss.item())

        if t % 5 == 0:
            with tf.stop_gradient():
                print_strategy(game.make_state().to(device))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (t + 1) % 10 == 0:
            print(f"Saving to ")
            tf.save(
                {
                    "epoch": t,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": path,
                },
                path,
            )
        if (t + 1) % 1000 == 0:
            tf.save(
                {
                    "epoch": t,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": path,
                },
                f"{path}.cp{t+1}",
            )


train()