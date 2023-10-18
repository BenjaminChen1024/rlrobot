# environment.py

import random

class CustomEnvironment:
    def __init__(self):
        self.state = 0
        self.action_space = [0, 1]
        self.state_space = [0, 1, 2, 3]

    def reset(self):
        self.state = random.choice(self.state_space)
        return self.state
    
    def step(self, action):
        # Define the transition dynamics based on environment
        if action == 0:
            if action == 0:
                reward = 0
                next_state = 0
            else:
                reward = 0
                next_state = 1

        self.state = next_state
        # Return the reward and next state
        return self.state, reward