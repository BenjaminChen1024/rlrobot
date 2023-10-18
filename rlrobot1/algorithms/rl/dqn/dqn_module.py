
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.n_observations = n_observations
        self.n_actions = n_actions

        self.layer1 = nn.Linear(self.n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, self.n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x