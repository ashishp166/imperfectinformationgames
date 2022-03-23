class Node:

    def __init__(self, actions, infoSet = ""):
        self.infoSet = infoSet
        self.actionCount = len(actions)
        self.clearStrategySum()
        self.regretSum = [0] * self.actionCount
        self.actions = actions

    def getStrategy(self, realizationWeight):
        strategy = [0] * self.actionCount
        normalizingSum = 0
        for a in range(self.actionCount):
            #if less than 0 then set to 0
            if self.regretSum[a] > 0:
                strategy[a] = self.regretSum[a]
            else: 
                strategy[a] = 0
            self.strategySum[a] += realizationWeight * strategy[a]

        for a in range(self.actionCount):
			if normalizingSum > 0:
				strategy[a] /= normalizingSum
			else:
				strategy[a] = 1.0 / self.actionCount
			self.strategySum[a] += realizationWeight * strategy[a]
        return strategy
    

    def getAverageStrategy(self):
        avgStrategy = [0] * self.actionCount
        normalizingSum = 0
        for a in range(self.actionCount):
            normalizingSum += self.strategySum[a]
        for a in range(self.actionCount):
            avgStrategy[a] = self.strategySum[a] / normalizingSum if normalizingSum > 0 else 1.0 / self.actionCount
        
        return avgStrategy
    
    def clearStrategySum(self):
        self.StrategySum = [0] * self.actionCount
    
    def getInfoSet(self):
        return self.infoSet
    
    def getRegretSum(self):
        return self.regretSum
    
    def getActions(self):
        return self.actions
    
    def getActionCount(self):
        return self.actionCount

