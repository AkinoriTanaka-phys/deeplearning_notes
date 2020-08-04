import numpy as np

key_b = [
    [' ', 'k', ' '],
    ['h', ' ', 'l'],
    [' ', 'j', ' '],
]

key2action = {
    'k':0,
    'j':1,
    'h':2,
    'l':3
    }

def show(board):
    for yoko in board:
        print(" ".join(yoko))

class Agent():
    def __init__(self, Policy):
        self.Policy = Policy
        
    def play(self):
        """
        return a number in [0,1,2,3] corresponding to [up, down, left, right]
        """
        return self.Policy.sample()
    
class Human(Agent):
    def __init__(self):
        self.Policy = 'Your policy'
        
    def play(self):
        show(key_b)
        while True:
            k = input()
            try:
                return key2action[k]
            except KeyError:
                print("type k/h/l/j")