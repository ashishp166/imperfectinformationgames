import random
import matplotlib.pyplot as plt
from counterfactual_regret_minimization import counterfactual_regret_minimization as cfr
import Node
from coin_toss import coin_toss

PASS = 0
BET = 1


def train(nodes, iterations, **options):
    game = coin_toss()
    util = 0
    utilOverTime = []
    doPlot = options.get("doPlot") == True

    for i in range(iterations):
        gameState = game.createState()
        util += cfr(nodes, game, gameState, 1, 1)

        # plot
        if doPlot:
            utilOverTime.append(util / (i + 1) if i > 10000 else None)
        # countdown
        if i % (iterations / 100) == 0 and i > 0:
            print(f"{i / (iterations / 100)}%\r", end="")
    print("Average game value for ", str(iterations), " runs: ", util / iterations)

    if doPlot:
        plt.plot(utilOverTime)
        plt.plot([-0.05555555] * len(utilOverTime))
        plt.show()

    print("Infoset\tPass\tBet" + " for " + str(iterations) + " runs")
    results = list(nodeMap.values())

    for n in results:
        l = list(enumerate(n.getAverageStrategy()))
        print(n.infoSet, '\t', list(map(lambda x: f'{n.actions[x[0]]}: {x[1]:0.4f}', l)))


def normalize(input):
    #get a percentage value for the strategy that is less than or equal to 1
    total = sum(input)
    if(total > 0):
        return [x / total for x in input]
    else:
        return [1.0/len(input) for x in input]
nodeMap = {}
train(nodeMap, 1000)  # Pretrain for the current strategy to converge a bit

for n in nodeMap.values():
    n.clearStrategySum()
train(nodeMap, 600000, doPlot=True)


