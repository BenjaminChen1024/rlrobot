# dqn_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class QNet(nn.Module):

    def __init__(self, num_observations, num_actions, hidden_size):
        super(QNet, self).__init__()

        self.num_observations = num_observations
        self.num_actions = num_actions

        self.layer1 = nn.Linear(self.num_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, self.num_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x