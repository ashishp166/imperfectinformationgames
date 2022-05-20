import random

import numpy as np

from LiarDieNode import Node

class LiarDieTrainer:
    def __init__(self):
        #doubt and accept should be static final variables
        self.DOUBT = 0
        self.ACCEPT = 1
        self.rand = random.random()
        self.sides
        self.responseNodes
        self.claimNodes

    def train(self, iterations):
        regrets = np.zeros(self.sides)
        rollAfterAcceptingClaim = np.zeros(self.sides, dtype=int)
        for iter in range(len(iterations)):
            #initialize rolls and start probabilities
            for i in range(len(rollAfterAcceptingClaim)):
                rollAfterAcceptingClaim[i] = random.nextInt(self.sides) + 1
            self.claimNodes[0][rollAfterAcceptingClaim[0]].pPlayer = 1
            self.claimNodes[0][rollAfterAcceptingClaim[0]].pOpponent = 1
            #accumulate realization weights forward
            for oppClaim in range(len(self.sides + 1)):
                #visit response nodes forward
                if oppClaim > 0:
                    for myClaim in range(oppClaim):
                        node = self.responseNodes[myClaim][oppClaim]
                        actionProb = node.getStrategy()
                        if(oppClaim < self.sides):
                            nextNode = self.claimNodes[oppClaim][rollAfterAcceptingClaim]
                            nextNode.pPlayer += actionProb[1] * node.pPlayer
                            nextNode.pOpponent += node.pOpponent
                #visit claim nodes forward
                if oppClaim < self.sides:
                    node = self.claimNodes[oppClaim][rollAfterAcceptingClaim[oppClaim]]
                    actionProb = node.getStrategy()
                    for myClaim in range(oppClaim + 1, len(self.sdies + 1)):
                        nextClaimProb = actionProb[myClaim - oppClaim - 1]
                        if nextClaimProb > 0:
                            nextNode = self.responseNodes[oppClaim][myClaim]
                            nextNode.pPlayer += node.pOpponent
                            nextNode.pOpponent += nextClaimProb * node.pPlayer

            #backpropagate utilites, adjusting regrets and strategies

            #reset strategy sums after half of training
    def LiarDieTrainer(self, sides):
        self.sides = sides
        self.responseNodes = Node[self.sides][self.sides + 1]
        for myClaim in range(len(self.sides + 1)):
            for oppClaim in range(myClaim + 1, len(self.sides + 1)):
                self.responseNodes[myClaim][oppClaim] = Node((oppClaim == 0 or oppClaim == self.sides) if 1 else 2)
        self.claimNodes = Node[self.sides][self.sides + 1]
        for oppClaim in range(len(self.sides)):
            for roll in range(1, len(self.sides + 1)):
                self.claimNodes[oppClaim][roll] = Node(self.sides - oppClaim)

