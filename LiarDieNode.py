import random

import numpy as np



class Node:
    def __init__(self, numActions):
        #these should be doubles, but currently are floats
        self.regretSum = np.zeros(numActions)
        self.strategy = np.zeros(numActions)
        self.strategySum = np.zeros(numActions)
        self.u = 0
        self.pPlayer = 0
        self.pOpponent = 0

    def getStrategy(self):
        normalizingSum = 0.0
        print(self.strategy)
        for a in range(len(self.strategy)):
            self.strategy[a] = max(self.regretSum[a], 0)
        for a in range(len(self.strategy)):
            if normalizingSum > 0:
                self.strategy[a] /= normalizingSum
            else:
                self.strategy[a] = 1.0/len(self.strategy)
        for a in range(len(self.strategy)):
            self.strategySum[a] += self.pPlayer * self.strategy[a]
        return self.strategy

    def getAverageStrategy(self):
        normalizingSum = 0.0
        for a in range(len(self.strategySum)):
            normalizingSum += self.strategySum[a]
        for a in range(len(self.strategy)):
            if normalizingSum > 0:
                self.strategySum[a] /= normalizingSum
            else:
                self.strategySum[a] = 1.0 / len(self.strategy)
        for a in range(len(self.strategy)):
            self.strategySum[a] = 1.0/len(self.strategySum)
        return self.strategySum

