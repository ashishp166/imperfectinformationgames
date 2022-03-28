from coin_toss_state import coin_toss_state
from Node import Node
import random

class coin_toss:

    CardTranslations = { 1: "h", 2: "t" }
    Player0Actions = [ "sell", "play" ]
    Player1Actions = [ "heads", "tails" ]

    def isTerminal(self, game_state):
        return (len(game_state.history) > 0 and game_state.history[-1] == 's') or len(game_state.history) > 1

    def getOutcome(self, game_state, player, opponent):
        lastActions = game_state.history[-1]
        outcome = coin_toss.CardTranslations[game_state.outcome]

        if lastActions == 's':
            if outcome == 'h':
                return 0.5
            else:
                return -0.5
        else:
            if lastActions == outcome:
                return -1
            else:
                return 1
    def applyGameEvents(self, gameState):
        if len(gameState.history) == 0:
            newState = gameState.copy()
            newState.outcome = self.toss()
            return newState
        else:
            return gameState
    def toss(self):
        faces = list(coin_toss.CardTranslations.keys())
        outcome = random.choice(faces)
        return outcome

    def createState(self):
        return coin_toss_state()

    def createNode(self, gameState):
        infoSet = self.getInfoSet(gameState)
        if self.getCurrentPlayer(gameState) == 0:
            return Node(coin_toss.Player0Actions, infoSet)
        else:
            return Node(coin_toss.Player1Actions, infoSet)
    def getInfoSet(self, gameState):
        player = self.getCurrentPlayer(gameState)
        if player == 1:
            return "Player 1 probability if Player 0 selects plays"
        if(str(coin_toss.CardTranslations[gameState.outcome]) == "h"):
            return "Player 0 probability for heads"
        else:
            return "Player 0 probability for tails"
    def getCurrentPlayer(self, gameState):
        plays = len(gameState.history)
        player = plays % 2
        return player
