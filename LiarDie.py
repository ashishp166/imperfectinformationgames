import random

import numpy as np

from LiarDieNode import Node

class LiarDieTrainer:
    def __init__(self, sides):
        #doubt and accept should be static final variables
        self.DOUBT = 0
        self.ACCEPT = 1
        self.rand = random.random()
        self.sides = sides
        self.responseNodes = np.empty((self.sides, self.sides + 1), dtype=Node)
        for myClaim in range(self.sides + 1):
            for oppClaim in range(myClaim + 1, self.sides + 1):
                self.responseNodes[myClaim][oppClaim] = Node(1 if oppClaim == 0 or oppClaim == self.sides else 2)
        self.claimNodes = np.empty((self.sides, self.sides + 1), dtype=Node)
        for oppClaim in range(self.sides):
            for roll in range(1, self.sides + 1):
                self.claimNodes[oppClaim][roll] = Node(self.sides - oppClaim)


    def train(self, iterations):
        regrets = np.zeros(self.sides)
        rollAfterAcceptingClaim = np.zeros(self.sides, dtype=int)
        for iter in range(iterations):
            #initialize rolls and start probabilities
            for i in range(len(rollAfterAcceptingClaim)):
                rollAfterAcceptingClaim[i] = random.randint(1, self.sides)
            self.claimNodes[0][rollAfterAcceptingClaim[0]].pPlayer = 1
            self.claimNodes[0][rollAfterAcceptingClaim[0]].pOpponent = 1
            #accumulate realization weights forward
            for oppClaim in range(self.sides + 1):
                #visit response nodes forward
                if oppClaim > 0:
                    for myClaim in range(oppClaim):
                        node = self.responseNodes[myClaim][oppClaim]
                        actionProb = node.getStrategy()
                        if(oppClaim < self.sides):
                            nextNode = self.claimNodes[oppClaim][rollAfterAcceptingClaim[oppClaim]]
                            nextNode.pPlayer += actionProb[1] * node.pPlayer
                            nextNode.pOpponent += node.pOpponent
                #visit claim nodes forward
                if oppClaim < self.sides:
                    node = self.claimNodes[oppClaim][rollAfterAcceptingClaim[oppClaim]]
                    actionProb = node.getStrategy()
                    for myClaim in range(oppClaim + 1, self.sides):
                        nextClaimProb = actionProb[myClaim - oppClaim - 1]
                        if nextClaimProb > 0:
                            nextNode = self.responseNodes[oppClaim][myClaim]
                            nextNode.pPlayer += node.pOpponent
                            nextNode.pOpponent += nextClaimProb * node.pPlayer

            #backpropagate utilites, adjusting regrets and strategies
            for oppClaim in range(self.sides, -1, -1):
                #visit claim nodes backwards
                if oppClaim < self.sides:
                    node = self.claimNodes[oppClaim][rollAfterAcceptingClaim[oppClaim]]
                    actionProb = node.strategy
                    node.u = 0.0
                    for myClaim in range(oppClaim + 1, self.sides + 1):
                        actionIndex = myClaim - oppClaim - 1
                        nextNode = self.responseNodes[oppClaim][myClaim]
                        childUtil = -nextNode.u
                        regrets[actionIndex] = childUtil
                        node.u += actionProb[actionIndex] * childUtil
                    for a in range(len(actionProb)):
                        regrets[a] -= node.u
                        node.regretSum[a] += node.pOpponent * regrets[a]
                    node.pPlayer = node.pOpponent = 0

                #visit response nodes backward
                if oppClaim > 0:
                    for myClaim in range(oppClaim):
                        node = self.responseNodes[myClaim][oppClaim]
                        actionProb = node.strategy
                        node.u = 0.0
                        doubtUtil = 1 if (oppClaim > rollAfterAcceptingClaim[myClaim]) else -1
                        regrets[self.DOUBT] = doubtUtil
                        if oppClaim < self.sides:
                            nextNode = self.claimNodes[oppClaim][rollAfterAcceptingClaim[oppClaim]]
                            regrets[self.ACCEPT] = nextNode.u
                            node.u += actionProb[self.DOUBT] * doubtUtil
                        for a in range(len(actionProb)):
                            regrets[a] -= node.u
                            node.regretSum[a] += node.pOpponent * regrets[a]
                        node.pPlayer = node.pOpponent = 0

            #reset strategy sums after half of training
            if iter == iterations / 2:
                for nodes in self.responseNodes:
                    for node in nodes:
                        if node is not None:
                            for a in range(len(node.strategySum)):
                                node.strategySum[a] = 0
                for nodes in self.claimNodes:
                    for node in nodes:
                        if node is not None:
                            for a in range(len(node.strategySum)):
                                node.strategySum[a] = 0
        #print resulting strategy
        for intialRoll in range(1, self.sides + 1):
            print("Initial claim policy with roll ", intialRoll)
            for prob in self.claimNodes[0][intialRoll].getAverageStrategy():
                print("{:.2f}".format(prob))

        print("\nOld Claim\tNew Claim\tAction Probabilities")
        for myClaim in range(self.sides + 1):
            for oppClaim in range(myClaim + 1, self.sides + 1):
                print("\t", myClaim, "\t", oppClaim, "\t",self.responseNodes[myClaim][oppClaim].getAverageStrategy(), "\n")

        print("\nOld Claim\tNew Claim\tAction Probabilities")
        for oppClaim in range(self.sides):
            for roll in range(1, self.sides + 1):
                print("\t", oppClaim, "\t", roll, "\t", self.claimNodes[oppClaim][roll].getAverageStrategy(),
                      "\n")



    def LiarDieTrainer(self, sides):
        self.sides = sides

