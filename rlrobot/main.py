# main.py

import numpy
import torch

from agent import DQNAgent, TRPOAgent
from environment import CustomEnvironment

# Initialize the environment
env = CustomEnvironment()
state_size = len(env.state_space)
action_size = len(env.action_space)

# Initialize the agent
# agent = DQNAgent(state_size, action_size)
agent = TRPOAgent(state_size, action_size)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state = torch.tensor([env.reset()], dtype=torch.float32)

    done = False

    while not done:
        action, action_prob = agent.selection_action(state)
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        old_probs.append(action_prob)

        state = new_state

    # Compute the advantage using GAE
    values = agent.value_network(torch.FloatTensor(states))
    advantages = agent.compute_advantage(rewards, values.detach().numpy())

    # Update the policy and value networks
    agent.update_policy(states, actions, advantages)
    agent.update_value(states, values)

print("Training finished")
