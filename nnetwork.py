# this is the neural network

import tensorflow as tf
import itertools
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

        # output size of each layer: 500, 400, 300, 200, 100, 1(tanh)
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
        # input is size of d_private
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
        # input size is in_channels
        self.conv = tf.keras.layers.Conv1D(filters=out_channels, kernel_size=kernel_size)

    # check over if the formatting of y is correct
    def forward(self, x, y=None):
        y = self.conv(y)


class Net3(tf.keras.Model):
    def __init__(self, d_pri, d_pub):
        super.__init__()

        def conv(channels, size):
            model = tf.keras.models.Sequential([
                # input size = 1
                tf.keras.layers.Conv1D(channels, kernel_size=size),
                # input size is channels
                tf.keras.layers.BatchNormalization(),
                tf.keras.relu(),
                # input size = channels
                tf.keras.layers.Conv1D(channels, kernel_size=size),
                # input size is channels
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
            # input size = channels
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


def calc_args(d1, d2, sides, variant):
    D_PUB = (d1 + d2) * sides
    if variant == "stairs":
        D_PUB = 2 * (d1 + d2) * sides

    LIE_ACTION = D_PUB
    D_PUB += 1

    N_ACTIONS = D_PUB

    CUR_INDEX = D_PUB
    D_PUB += 1

    D_PUB_PER_PLAYER = D_PUB
    D_PUB *= 2

    D_PRI = max(d1, d2) * sides

    PRI_INDEX = D_PRI
    D_PRI += 2

    return D_PUB, D_PRI, N_ACTIONS, LIE_ACTION, CUR_INDEX, PRI_INDEX, D_PUB_PER_PLAYER


class Game:
    def __init__(self, model, d1, d2, sides, variant):
        self.model = model
        self.D1 = d1
        self.D2 = d2
        self.SIDES = sides
        self.VARIANT = variant

        (
            self.D_PUB,
            self.D_PRI,
            self.N_ACTIONS,
            self.LIE_ACTION,
            self.CUR_INDEX,
            self.PRI_INDEX,
            self.D_PUB_PER_PLAYER,
        ) = calc_args(d1, d2, sides, variant)

    def make_regrets(self, priv, state, last_call):

        if priv[self.PRI_INDEX] != state[self.CUR_INDEX]:
            print("Warning: Regrets are not with respect to current player")

        n_actions = self.N_ACTIONS - last_call - 1

        batch = state.repeat(n_actions + 1, 1)

        for i in range(n_actions):
            self._apply_action(batch[i + 1], i + last_call + 1)

        priv_batch = priv.repeat(n_actions + 1, 1)

        v, *vs = list(self.model(priv_batch, batch))
        return [max(vi - v, 0) for vi in vs]
        # alternate way of hedging
        # return [math.exp(10*(vi - v)) for vi in vs]

    def evaluate_call(self, r1, r2, last_call):

        if last_call == -1:
            return True

        n, d = divmod(last_call, self.SIDES)
        n, d = n + 1, d + 1

        cnt = Counter(r1 + r2)
        if self.VARIANT == "normal":
            actual = cnt[d]
        if self.VARIANT == "joker":
            actual = cnt[d] + cnt[1] if d != 1 else cnt[d]
        if self.VARIANT == "stairs":
            if all(r == i + 1 for r, i in zip(r1, range(self.SIDES))):
                actual += 2 * len(r1) - r1.count(d)
            if all(r == i + 1 for r, i in zip(r2, range(self.SIDES))):
                actual += 2 * len(r2) - r1.count(d)
        return actual >= n

    def policy(self, priv, state, last_call, eps=0):
        regrets = self.make_regrets(priv, state, last_call)
        for i in range(len(regrets)):
            regrets[i] += eps
        if sum(regrets) <= 0:
            return [1 / len(regrets)] * len(regrets)
        else:
            s = sum(regrets)
            return [r / s for r in regrets]

    def sample_action(self, priv, state, last_call, eps):
        pi = self.policy(priv, state, last_call, eps)
        action = next(iter(tf.utils.data.WeightedRandomSampler(pi, num_samples=1)))
        return action + last_call + 1

    def apply_action(self, state, action):
        new_state = state.clone()
        self._apply_action(new_state, action)
        return new_state

    def _apply_action(self, state, action):
        # Inplace
        cur = self.get_cur(state)
        state[action + cur * self.D_PUB_PER_PLAYER] = 1
        state[self.CUR_INDEX + cur * self.D_PUB_PER_PLAYER] = 0
        state[self.CUR_INDEX + (1 - cur) * self.D_PUB_PER_PLAYER] = 1
        return state

    def make_priv(self, roll, player):
        assert player in [0, 1]
        if (self.D_PRI % 12 != 0):
            priv = tf.Variable(tf.zeros([24]))
        else:
            priv = tf.Variable(tf.zeros([self.D_PRI]))
        priv = priv[self.PRI_INDEX + player].assign(1)
        cnt = Counter(roll)
        for face, c in cnt.items():
            for i in range(c):
                priv = priv[(face - 1) * max(self.D1, self.D2) + i].assign(1)
        return priv

    def make_state(self):

        state = tf.Variable(tf.zeros([self.D_PUB]))
        state = state[self.CUR_INDEX].assign(1)
        return state

    def get_cur(self, state):
        # one hot encoding of other player
        return 1 - int(state[self.CUR_INDEX])

    def rolls(self, player):
        assert player in [0, 1]
        n_faces = self.D1 if player == 0 else self.D2
        return [
            tuple(sorted(r))
            for r in itertools.product(range(1, self.SIDES + 1), repeat=n_faces)
        ]

    def get_calls(self, state):
        merged = (
                state[: self.CUR_INDEX]
                + state[self.D_PUB_PER_PLAYER: self.D_PUB_PER_PLAYER + self.CUR_INDEX]
        )
        return (merged == 1).nonzero(as_tuple=True)[0].tolist()

    def get_last_call(self, state):
        ids = self.get_calls(state)
        if not ids:
            return -1
        return int(ids[-1])

