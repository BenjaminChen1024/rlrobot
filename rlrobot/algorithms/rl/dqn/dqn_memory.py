# dqn_memory.py

import torch

import random
from collections import namedtuple, deque
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


"""
class RolloutStorage:

    def __init__(self, num_transitions, observations_shape, states_shape, actions_shape, device='cpu', sampler='sequential'):
        
        self.device = device
        self.sampler = sampler

        self.observations = torch.zeros(num_transitions, *observations_shape, device=self.device)
        self.states = torch.zeros(num_transitions, *states_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions, 1, device=self.device)
        self.actions = torch.zeros(num_transitions, *actions_shape, device=self.device)
        
        self.num_transitions = num_transitions
        self.step = 0


    def push(self, observations, states, actions, rewards):

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.states].copy_(rewards)

        self.step += 1

    def sample(self, batch_size):
        subset = SubsetRandomSampler(range(batch_size))
        batch = BatchSampler(subset, batch_size, drop_last=True)
        return batch

    def __len__(self):
        return len(self.memory)

"""
