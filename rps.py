import numpy as np
import random

import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

ROCK, PAPER, SCISSORS = 0, 1, 2
NUM_ACTIONS = 3
# this is the opponent strategy and this means they favor rock so player
# should favor paper to win
oppStrategy = np.array([0.5, 0.25, 0.25])


def value(p1, p2):
    if p1 == p2:
        return 0
    elif (p1 - 1) % NUM_ACTIONS == p2:
        return 1
    else:
        return -1


def normalize(strategy):
    strategy = np.copy(strategy)
    normalizingSum = np.sum(strategy)
    if normalizingSum > 0:
        strategy /= normalizingSum
    else:
        strategy = np.ones(strategy.shape[0]) / strategy.shape[0]
    return strategy

def getStrategy(regretSum):
    return normalize(np.maximum(regretSum, 0))

def getAveragedStrategy(strategySum):
    return normalize(strategySum)

def getAction(strategy):
    strategy = strategy / np.sum(strategy) #normalize
    return np.searchsorted(np.cumsum(strategy), random.random())

def train(iterations):
    regretSum = np.zeros(NUM_ACTIONS)
    strategySum = np.zeros((NUM_ACTIONS))

    actionUtility = np.zeros(NUM_ACTIONS)
    for i in range(iterations):
        strategy = getStrategy(regretSum)
        #use the regret to form the current player strategy
        strategySum += strategy

        myAction = getAction(strategy)
        otherAction = getAction(oppStrategy)

        #this is snaking around the values so that it sums up the new regret value to proper item
        actionUtility[otherAction] = 0
        actionUtility[(otherAction + 1) % NUM_ACTIONS] = 1
        actionUtility[(otherAction - 1) % NUM_ACTIONS] = -1

        regretSum += actionUtility - actionUtility[myAction]
        return strategySum

strategySum = train(100000)

strategy = getAveragedStrategy(strategySum)
print(strategy)

vvv = []
for j in range(200):
    vv = 0
    for i in range(100):
        #strategy = getStrategy()
        #strategy = [0, 1, 0]

        myAction = getAction(strategy)
        otherAction = getAction(oppStrategy)
        vv += value(myAction, otherAction)
    vvv.append(vv)
plt.plot(sorted(vvv))
print(np.mean(vvv))
print(np.median(vvv))
plt.show()

