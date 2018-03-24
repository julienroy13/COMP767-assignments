import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import utils
from model import MLP, CNN

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

    # Instantiating the model
    model_type = config.get('model_type', 'MLP')
    if model_type == "MLP":
        model = MLP(config['input_size'], config["hidden_layers"], config['output_size'], config["nonlinearity"], config["initialization"], config["dropout"], verbose=True)
    elif model_type == "CNN":
        model = CNN(config["initialization"], config["is_batch_norm"], verbose=True)
    else:
        raise ValueError('config["model_type"] not supported : {}'.format(model_type))

    if config['resume']:
        model.load_state_dict(torch.load(os.path.join(save_dir, exp_name, "model")))

    # If GPU is available, sends model and dataset on the GPU
    if torch.cuda.is_available() and config["use_cuda"]:
        model.cuda(gpu_id)
        print("USING GPU-{}".format(gpu_id))

    # Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.NLLLoss()

    env = gym.make("MountainCar-v0")
    observation = env.reset()

    LL_list, reward_list = [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0

    start = time.time()

    # Initializing recorders
    update = 0
    loss_tape = []
    episode_lengths = []

    # TRAINING LOOP
    while update < config['max_updates']:

        if config['render']: 
            screen = env.render(mode='rgb_array')
            plt.imsave("test.png", env.render(mode='rgb_array'))

        x = Variable(torch.from_numpy(observation).float(), requires_grad=False)
        
        if torch.cuda.is_available() and config["use_cuda"] :
            x = x.cuda(gpu_id)

        # Feedforward through the policy network
        action_prob = torch.unsqueeze(model(x), 0)
        
        # Sample an action from the returned probability using epsilon-greedy policy
        if np.random.uniform() < config['epsilon']:
            action = Variable(torch.IntTensor(np.random.randint(low=0, high=3, size=(1,)))).long()
        else:
            action = torch.max(action_prob, 1)[1].long()

        if torch.cuda.is_available() and config["use_cuda"] :
            action = action.cuda(gpu_id)

        # record the log-likelihoods
        NLL = loss_fn(action_prob, action)
        LL_list.append(NLL)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(int(action.cpu().data.numpy()))
        reward_sum += reward

        reward_list.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if done: # an episode finished (an episode ends when one of the player wins 21 games)
            episode_number += 1

            # Computes loss and reward for each step of the episode
            R = torch.zeros(1, 1)
            loss = 0
            for i in reversed(range(len(reward_list))):
                R = config['gamma'] * R + reward_list[i]
                Return_i = Variable(R)
                if torch.cuda.is_available() and config["use_cuda"] :
                    Return_i = Return_i.cuda(gpu_id)
                loss = loss + (LL_list[i] * Return_i).squeeze()
            loss = loss / len(reward_list)

            # Backpropagates to compute the gradients
            loss.backward()

            # Performs parameter update every config['mb_size'] episodes
            if episode_number % config['mb_size'] == 0:

                # Takes one training step
                optimizer.step()
                    
                # Empties the gradients
                optimizer.zero_grad()

                stop = time.time()
                print("PARAMETER UPDATE ------------ Time takes : {}s".format(int(stop - start)))
                start = time.time()

                if config['save_plots']:
                    utils.save_results_MOUNTAINCAR(save_dir, exp_name, loss_tape, episode_lengths, config)
                update += 1
                if update % 10 == 0: 
                    torch.save(model.state_dict(), os.path.join(save_dir, exp_name, "model_"+model.name()))
                
            # Records the average loss and score of the episode
            loss_tape.append(loss.cpu().data.numpy())
            episode_lengths.append(len(reward_list))

            LL_list, reward_list = [], [] # reset array memory

            # More book-keeping
            if running_reward is None:
                running_reward = reward_sum
            else:
                running_reward = running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was {0:.2f}. running mean: {1:.2f}'.format(reward_sum, running_reward))
            
            reward_sum = 0
            observation = env.reset() # reset env


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
