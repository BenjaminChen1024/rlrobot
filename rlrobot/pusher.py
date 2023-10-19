# pusher.py

# Import envs
import gymnasium as gym
# Import torch
import torch
import torch.nn as nn
# Import self module
from rlrobot.algorithms.rl.dqn.dqn import DQN

# If GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create pusher envs
env = gym.make("Pusher-v4", render_mode="human")
# Creatre algorithm
agent = DQN(env)

# Training loop
num_episodes = 600
agent.run(num_episodes)
