import numpy as np
import random

import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

PASS, BET, NUM_ACTIONS = 0, 1, 2

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

def innertrain(regretSum, strategySum, oppStrategy):
    #accmulate the current strategy based on regret
    strategy = getStrategy(regretSum)
    strategySum += strategy

    #select my action and opponent action
    myAction = getAction(strategy)
    otherAction = getAction(oppStrategy)

    actionUtility = np.zeros(NUM_ACTIONS)
    # this is snaking around the values so that it sums up the new regret value to proper item
    actionUtility[otherAction] = 0
    actionUtility[(otherAction + 1) % NUM_ACTIONS] = 1
    actionUtility[(otherAction - 1) % NUM_ACTIONS] = -1

    regretSum += actionUtility - actionUtility[myAction]
    return regretSum, strategySum

def train(iterations):
    regretSum = np.zeros(NUM_ACTIONS)
    strategySum = np.zeros(NUM_ACTIONS)
    oppStrategy = np.array([0.4, 0.3, 0.3])

    for i in range(iterations):
        regretSum, strategySum = innertrain(regretSum, strategySum, oppStrategy)
    return strategySum

def train2p(oiterations, iterations):
    strategySumP1 = np.zeros(NUM_ACTIONS)
    strategySumP2 = np.zeros(NUM_ACTIONS)

    for j in range(oiterations):
        oppStrategy = normalize(strategySumP2)
        regretSumP1 = np.zeros(NUM_ACTIONS)
        for i in range(iterations):
            regretSumP1, strategySumP1 = innertrain(regretSumP1, strategySumP1, oppStrategy)

        oppStrategy = normalize(strategySumP1)
        regretSumP2 = np.zeros(NUM_ACTIONS)
        for i in range(iterations):
            regretSumP2, strategySumP2 = innertrain(regretSumP2, strategySumP2, oppStrategy)
        print(normalize(strategySumP1), normalize(strategySumP2))
    return strategySumP1, strategySumP2

s1, s2 = train2p(20, 1000)
playerstrategey = normalize(s1)
oppStrategy = normalize(s2)
print(playerstrategey)
print(oppStrategy)

comp = train(10000)
print(getAveragedStrategy(comp))

vvv = []
for j in range(200):
    vv = 0
    for i in range(100):
        myAction = getAction(playerstrategey)
        otherAction = getAction(oppStrategy)
        vv += value(myAction, otherAction)
    vvv.append(vv)
plt.plot(sorted(vvv))
plt.show()
print(np.mean(vvv))
print(np.median(vvv))


strategy = getAveragedStrategy(s1)
for j in range(5):
    a = [getAction(getStrategy(strategy)) for i in range(100)]
    plt.plot(sorted(a))
plt.show()