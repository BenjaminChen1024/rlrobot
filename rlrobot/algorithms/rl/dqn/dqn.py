# dqn.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from rlrobot.algorithms.rl.dqn.dqn_module import QNet
from rlrobot.algorithms.rl.dqn.dqn_memory import ReplayMemory
from collections import namedtuple, deque

import matplotlib.pyplot as plt
from itertools import count
import random
import math
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN:
    def __init__(self, env, device='cpu'):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.device = device
        self.memory = ReplayMemory(100000)

        # DQN parameters
        self.learning_rate = 1e-4
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.005

        self.eps = 1e-6
        self.steps_done = 0
        self.initial_std = 0.3

        # DQN components
        self.env = env
        self.policy_net = QNet(self.observation_space.shape[0], self.action_space.shape[0]).to(self.device)
        self.target_net = QNet(self.observation_space.shape[0], self.action_space.shape[0]).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Action noise
        self.log_std = nn.Parameter(np.log(self.initial_std) * torch.ones(self.action_space.shape))

        self.episode_durations = []

    """
    def act(self, state):
        action_mean = self.policy_net(state)
        
        convariance = self.log_std.exp() * self.log_std.exp()
        distribution = Normal(action_mean[0] + self.eps, convariance)
        action = distribution.sample()
        action_log_prob = distribution.log_prob(action)

        action = action.numpy()

        return action
    """

    def act(self,state):
        sample = random.random()
        eps_threshold = 0.05 + (0.9 - 0.05) * math.exp(-1.* self.steps_done / 1000)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():

                print(state)
                return self.policy_net(state)
        else:
            return self.action_space.sample()
    
    def log(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99),means))
            plt.plot(means.numpy())

        plt.pause(0.001)

    
    def update(self):
        if len(self.memory) < self.batch_size:
                return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = batch.reward
        print(state_batch)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

        

    def run(self, num_episodes):
        for i in range(num_episodes):
            # Initialize the environment and get state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype = torch.float32, device=self.device)
            for t in count():
                # Compute the action
                action = self.act(state)
                action = torch.tensor(action, dtype=torch.float32, device=self.device)

                # Step the environment
                observation, reward, terminated, truncated, info = self.env.step(action)
                reward = torch.tensor(reward, dtype = torch.float32, device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device)
                
                # Store the transition in memort
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                self.update()

                if done:
                    self.episode_durations.append(t+1)
                    self.log()
                    break

        print('Complete')
        self.log(show_result=True)
        plt.ioff()
        plt.show()  
                



