import torch
import torch.nn as nn
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
from gym import wrappers

from termcolor import cprint


def train_model(config, gpu_id, save_dir, exp_name):

    env = gym.make(config['env_name'])

    env.seed(123)
    torch.manual_seed(123)
    np.random.seed(123)

    agent = REINFORCE(len(env.observation_space.sample()), config['hidden_layers'], env.action_space.n, config['lr'], config['use_cuda'], gpu_id)

    if config['resume']:
        agent.load_policy(directory=os.path.join(save_dir, exp_name))

    # TRAINING LOOP
    episode_number = 0
    running_average = None
    loss_tape, episode_lengths = [], []
    while episode_number < config['max_episodes']:
        
        # Book Keeping
        episode_number += 1
        observation = env.reset()
        NLL_list, reward_list = [], []
        agent.reset_counters()
        
        done = False
        t = 0
        # RUN ONE EPISODE
        while not(done) and t < config['max_steps']:
            action, log_prob = agent.select_action(observation)
            observation, reward, done, _ = env.step(action)

            if config['env_name'] == "MountainCar-v0":
                done = bool(observation[0] >= 0.5)

            if config['render']: 
                env.render()
                
            if episode_number in config['video_ckpt']:
                image = env.render(mode='rgb_array')
                video_folder = os.path.join(save_dir, exp_name, "video_ckpts".format(episode_number))
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)
                plt.imsave(os.path.join(video_folder, "ep{}_{}.png".format(episode_number, t)), image)

            NLL_list.append(log_prob)
            reward_list.append(reward)
            t += 1

        # UPDATE THE PARAMETERS (for Monte-Carlo method)
        loss = agent.compute_gradients(reward_list, NLL_list, config['gamma'])
        agent.update_parameters()

        # More book-keeping
        loss_tape.append(loss)
        episode_lengths.append(len(reward_list))
        if running_average is None:
            running_average = np.sum(reward_list)
        else:
            running_average = running_average * 0.9 + np.sum(reward_list) * 0.1
        print("Episode: {}, reward: {}, average: {:.2f}".format(episode_number, np.sum(reward_list), running_average))

        if episode_number % config['chkp_freq'] == 0:
            agent.save_policy(directory=os.path.join(save_dir, exp_name))
            utils.save_results_classicControl(save_dir, exp_name, loss_tape, episode_lengths, config)
            
    env.close()


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