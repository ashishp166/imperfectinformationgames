from Node import Node

def cfr(nodes, game, gameState, p0, p1):
    #Player determination
    player = game.getCurrentPlayer(gameState)

    #game events
    gameState = game.applyGameEvents(gameState)

    if game.isTerminal(gameState):
        opponent = 1 - player
        return game.getOutcome(gameState, player, opponent)
    

    infoSet = game.getInfoSet(gameState)
    node = None
    if not infoSet in nodes:
        noe = game.createNode(gameState)
        nodes[infoSet] = node
    else:
        node = nodes[infoSet]
    

    realizationWeight = [p0, p1][player]
    strategy = node.getStrategy(realizationWeight)
    util = [0] * node.getActionCount()
    nodeUtil = 0
    for a in range(node.getActionCount()):
        newState = gameState.nextHistory(a)
        util[a] = -1.0 * cfr(ndoes, game, newState, [p0 * strategy[a], p0][player], [p1, p1 * strategy[a]][player])
        nodeUtil += strategy[a] * util[a]
    

    for a in range(node.getActionCount()):
        regret = util[a] - nodeUtil
        node.regret[a] += [p1, p0][player] * regret
    

    return nodeUtil