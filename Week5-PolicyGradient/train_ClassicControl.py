import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import utils
from model import MLP
from agent import REINFORCE

from configs import myConfigs
import os
import argparse
import math
import time
import pdb
import matplotlib.pyplot as plt

import numpy as np
import gym

from termcolor import cprint

torch.manual_seed(1234)


def train_model(config, gpu_id, save_dir, exp_name):

    torch.manual_seed(123)
    np.random.seed(123)

    # Instanciates the gym-environment
    env = gym.make(config['env'])
    observation = env.reset()
    observation = utils.normalize_observation(observation, env)

    # Instanciates the agent
    agent = REINFORCE(len(env.observation_space.sample()), config['hidden_layers'], env.action_space.n, config['lr'], config['use_cuda'], gpu_id)

    if config['resume']:
        agent.load_policy(directory=os.path.join(save_dir, exp_name))

    # Book Keeping
    NLL_list, reward_list = [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0
    update = 0
    loss_tape = []
    episode_lengths = []

    start = time.time()

    # TRAINING LOOP
    while update < config['max_updates']:

        if config['render']: 
            screen = env.render(mode='rgb_array')
            #plt.imsave("test.png", env.render(mode='rgb_array'))

        # Action selection
        action, NLL = agent.select_action(observation)
        NLL_list.append(NLL)

        # Step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        observation = utils.normalize_observation(observation, env)
        reward_sum += reward

        reward_list.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        # If the episode is over
        if done:
            episode_number += 1

            # Updates the agent's policy based on previous episode
            loss = agent.compute_gradients(reward_list, NLL_list, config['gamma'])

            # Performs parameter update every config['mb_size'] episodes
            if episode_number % config['mb_size'] == 0:

                # Takes one training step
                agent.update_parameters()

                stop = time.time()
                print("PARAMETER UPDATE ------------ Time takes : {}s".format(int(stop - start)))
                start = time.time()

                if config['save_plots']:
                    utils.save_results_classicControl(save_dir, exp_name, loss_tape, episode_lengths, config)
                
                update += 1
                if update % 10 == 0: 
                    agent.save_policy(directory=os.path.join(save_dir, exp_name))
                
            # Records the average loss and score of the episode
            loss_tape.append(loss)
            episode_lengths.append(len(reward_list))

            NLL_list, reward_list = [], [] # reset array memory

            # More book-keeping
            if running_reward is None:
                running_reward = reward_sum
            else:
                running_reward = running_reward * 0.99 + reward_sum * 0.01
            print('Episode {0} over \n\tTotal reward: {1:.2f} \n\tRunning mean: {2:.2f} \n\tAction count: {3}\n\tExplore-Exploit: ({4} ; {5})'.format(episode_number, reward_sum, running_reward, agent.action_count, agent.explore_count, agent.exploit_count))
            
            reward_sum = 0
            agent.reset_counters()
            observation = env.reset() # reset env
            observation = utils.normalize_observation(observation, env)


if __name__ == "__main__":
        # Retrieves arguments from the command line
        parser = argparse.ArgumentParser()

        parser.add_argument('--config', type=str, default='0',
                                                help='config id number')

        parser.add_argument('--gpu', type=str, default='0',
                                                help='gpu id number')

        args = parser.parse_args()
        print(args)

        # Extracts the chosen config
        config_number = int(args.config)
        config = myConfigs[config_number]
        gpu_id = int(args.gpu)
        save_dir = "results"
        exp_name = "config" + str(config_number)

        if not os.path.exists(os.path.join(save_dir, exp_name)):
            os.makedirs(os.path.join(save_dir, exp_name))

        # Runs the training procedure
        print("Running the training procedure for config-{}".format(config_number))
        train_model(config, gpu_id, save_dir, exp_name)
