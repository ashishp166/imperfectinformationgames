class coin_toss_state:
    def __init__(self, outcome=None, history=""):
        self.outcome = outcome
        self.history = history

    def nextHistory(self, action):
        nextStep = "X"
        if len(self.history) == 0:
            nextStep = ["o", "p"][action]
        else:  # len(self.history) == 1:
            nextStep = ["h", "t"][action]

        newState = coin_toss_state(self.outcome, self.history + nextStep)
        return newState

    def copy(self):
        newState = coin_toss_state(self.outcome, self.history)
        return newState
