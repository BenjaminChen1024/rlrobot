# ppo.py

import torch
import torch.nn as nn

import os
from datetime import datetime
import gymnasium as gym
import numpy as np

from rlrobot.algorithms.rl.ppo.ppo_memory import RolloutBuffer
from rlrobot.algorithms.rl.ppo.ppo_module import ActorCritic

class PPO:
    def __init__(self, env_name, has_continuous_action_space, device='cuda'):

        # Initialize environment hyperparameters
        self.env_name = env_name
        self.has_continuous_action_space = has_continuous_action_space
        self.device= device

        self.max_ep_len = 1000                   # max timesteps in one episode
        self.max_training_timesteps = int(1e6)   # break training loop if timeteps > max_training_timesteps
        self.total_test_episodes = 100    # total num of testing episodes

        self.print_freq = self.max_ep_len * 10        # print avg reward in the interval (in num timesteps)
        self.log_freq = self.max_ep_len * 2           # log avg reward in the interval (in num timesteps)
        self.save_model_freq = int(1e5)          # save model frequency (in num timesteps)

        self.action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
        self.action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        self.min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
        self.action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)



        # PPO hyperparameters
        self.update_timestep = self.max_ep_len * 4    # update policy every n timesteps
        self.K_epochs = 80                  # update policy for K epochs in one PPO update
        self.eps_clip = 0.2                 # clip parameter for PPO
        self.gamma = 0.99                   # discount factor
        self.lr_actor = 0.0003              # learning rate for actor network
        self.lr_critic = 0.001              # learning rate for critic network
        self.random_seed = 0                # set random seed if required (0 = no random seed)

        # Create environment
        self.env = gym.make(self.env_name, render_mode='human')
        # State space dimension
        self.state_dim = self.env.observation_space.shape[0]
        # Action space dimension
        if self.has_continuous_action_space:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(self.state_dim, self.action_dim, self.has_continuous_action_space, self.action_std, self.device).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
                    ])

        self.policy_old = ActorCritic(self.state_dim, self.action_dim, self.has_continuous_action_space, self.action_std, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.tensor(state[0], dtype=torch.float32).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    
    def log(self):
        # log files for multiple runs are NOT overwritten
        log_dir = "log/PPO_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + self.env_name + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        #  get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)

        # create new log file for each run
        self.log_f_name = log_dir + '/PPO_' + self.env_name + "_log_" + str(run_num) + ".csv"

        print("current logging run number for " + self.env_name + " : ", run_num)
        print("logging at : " + self.log_f_name)

        # checkpointing 
        self.run_num_pretrained = 0      # change this to prevent overwriting weights in same env_name folder

        directory = "log/PPO_preTrained"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/' + self.env_name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(self.env_name, self.random_seed, self.run_num_pretrained)
        print("save checkpoint path : " + self.checkpoint_path)

    def print(self):
        print("--------------------------------------------------------------------------------------------")
        print("max training timesteps : ", self.max_training_timesteps)
        print("max timesteps per episode : ", self.max_ep_len)
        print("model saving frequency : " + str(self.save_model_freq) + " timesteps")
        print("log frequency : " + str(self.log_freq) + " timesteps")
        print("printing average reward over episodes in last : " + str(self.print_freq) + " timesteps")
        print("--------------------------------------------------------------------------------------------")
        print("state space dimension : ", self.state_dim)
        print("action space dimension : ", self.action_dim)
        print("--------------------------------------------------------------------------------------------")
        if has_continuous_action_space:
            print("Initializing a continuous action space policy")
            print("--------------------------------------------------------------------------------------------")
            print("starting std of action distribution : ", self.action_std)
            print("decay rate of std of action distribution : ", self.action_std_decay_rate)
            print("minimum std of action distribution : ", self.min_action_std)
            print("decay frequency of std of action distribution : " + str(self.action_std_decay_freq) + " timesteps")
        else:
            print("Initializing a discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("PPO update frequency : " + str(self.update_timestep) + " timesteps")
        print("PPO K epochs : ", self.K_epochs)
        print("PPO epsilon clip : ", self.eps_clip)
        print("discount factor (gamma) : ", self.gamma)
        print("--------------------------------------------------------------------------------------------")
        print("optimizer learning rate actor : ", self.lr_actor)
        print("optimizer learning rate critic : ", self.lr_critic)
        if self.random_seed:
            print("--------------------------------------------------------------------------------------------")
            print("setting random seed to ", self.random_seed)
            torch.manual_seed(self.random_seed)
            self.env.seed(self.random_seed)
            np.random.seed(self.random_seed)
        
    def train(self):
        self.log()
        self.print()


        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print("=========================================================================")

        # logging file
        log_f = open(self.log_f_name,"w+")
        log_f.write('episode,timestep,reward\n')

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        # training loop
        while time_step <= self.max_training_timesteps or print_avg_reward >= -50:

            state = self.env.reset()
            state = state[0]
            current_ep_reward = 0

            for t in range(1, self.max_ep_len+1):

                # select action with policy
                action = self.select_action(state)
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # saving reward and is_terminals
                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)

                time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if time_step % self.update_timestep == 0:
                    self.update()

                # if continuous action space; then decay action std of ouput action distribution
                if has_continuous_action_space and time_step % self.action_std_decay_freq == 0:
                    self.decay_action_std(self.action_std_decay_rate, self.min_action_std)

                # log in logging file
                if time_step % self.log_freq == 0:

                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % self.print_freq == 0:

                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % self.save_model_freq == 0:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + self.checkpoint_path)
                    self.save(self.checkpoint_path)
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("----------------------------------------------------------------------------")

                # break; if the episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1
            
        log_f.close()
        self.env.close()

        # print total training time
        print("=========================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("=========================================================================")

    def test(self):
        self.log()

        directory = "log/PPO_preTrained" + '/' + env_name + '/'
        checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, self.random_seed, self.run_num_pretrained)
        print("loading network from : " + checkpoint_path)

        self.load(checkpoint_path)

        print("-------------------------------------------------------------------------")

        test_running_reward = 0

        for ep in range(1, self.total_test_episodes+1):
            ep_reward = 0
            state = self.env.reset()
            state = state[0]

            for t in range(1, self.max_ep_len+1):
                action = self.select_action(state)
                state, reward, terminated, truncated, info = self.env.step(action)
                done = truncated

                ep_reward += reward

                if done:
                    break

            # clear buffer
            self.buffer.clear()

            test_running_reward +=  ep_reward
            print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
            ep_reward = 0

        self.env.close()

        print("==========================================================================")

        avg_test_reward = test_running_reward / self.total_test_episodes
        avg_test_reward = round(avg_test_reward, 2)
        print("average test reward : " + str(avg_test_reward))

        print("==========================================================================")


if __name__ == '__main__':
    # Set device to cpu or cuda
    device = torch.device('cpu')
    if(torch.cuda.is_available()): 
        device = torch.device('cuda') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    # Create envs
    env_name = "Pusher-v4"
    has_continuous_action_space=True
    print("Training environment name : " + env_name)
    # Creatre agent
    agent = PPO(env_name, has_continuous_action_space, device)
    # Train agent
    # agent.train()
    # Test agent
    agent.test()
    
    
    
    
    
    
    



