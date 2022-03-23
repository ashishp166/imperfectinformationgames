from statistics import mean
import random

CHOICES = 3
ROCK = 0
PAPER = 1
SCISSORS = 2

ITERATIONS = 10**5

def nonNegatify(input):
    #goal is to set all negative numbers to 0
    return [x if x > 0 else 0 for x in input]


def normalize(input):
    #get a percentage value for the strategy that is less than or equal to 1
    total = sum(input)
    if(total > 0):
        return [x / total for x in input]
    else:
        return [1.0/len(input) for x in input]


def new_strategy(opp):
    return normalize(nonNegatify(opp))

def get_action(curr):
    rand = random.random()
    cumilativeProbability = 0
    for i in range(CHOICES):
        cumilativeProbability = cumilativeProbability + curr[i]
        if rand < cumilativeProbability:
            return i
    
    return CHOICES - 1


def get_object(object):
    #just flip the values that were listed in the rps values
    if object == ROCK:
        return [1, 2, 0]
    elif object == PAPER:
        return [0, 1, 2]
    else: #SCISSORS
        return [2, 0, 1]

def run(p, o):
    opponent = new_strategy(o)
    prob = get_action(opponent)
    obj = get_object(prob)
    p = [sum(x) for x in zip(obj, p)]
    return p

def train(player, opponent):
    for i in range(ITERATIONS):
        player = run(player, opponent)
        opponent = run(opponent, player)
    return [player, opponent]

print("Start (rock, paper, scissor)")
p_regrets = [0.0] * CHOICES
p_regrets[0] = 100

o_regrets = [0.0] * CHOICES
o_regrets[1] = 100

val = train(p_regrets, o_regrets)

print("player percentages ", normalize(val[0]))
print("opp percentages ", normalize(val[1]))

print("Done")