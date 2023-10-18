# agent.py

import random
import torch
import torch.optim as optim
from rlrobot.network import DQN
from environment import CustomEnvironment

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=64, gamma=0.99, epsilon=1, epsilon_min=0.01, epsilon_decay=0.9999, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.memory = []
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.tensor(state, dtype=torch.float)
            state = state.unsqueeze(0)
            actions = self.model(state)
            return torch.argmax(actions).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state
