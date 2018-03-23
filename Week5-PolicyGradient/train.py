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

import numpy as np
import gym

from termcolor import cprint
import psutil
process = psutil.Process(os.getpid())

torch.manual_seed(1234)


def train_model(config, gpu_id, save_dir, exp_name):

    # Instantiating the model
    model_type = config.get('model_type', 'MLP')
    if model_type == "MLP":
        model = MLP(config['input_size'], config["hidden_layers"], 1, config["nonlinearity"], config["initialization"], config["dropout"], verbose=True)
    elif model_type == "CNN":
        model = CNN(config["initialization"], config["is_batch_norm"], verbose=True)
    else:
        raise ValueError('config["model_type"] not supported : {}'.format(model_type))

    if config['resume']:
        model.load_state_dict(torch.load(os.path.join(save_dir, exp_name, "model")))

    # If GPU is available, sends model and dataset on the GPU
    if torch.cuda.is_available():
        model.cuda(gpu_id)
        print("USING GPU-{}".format(gpu_id))

    # Optimizer and Loss Function
    optimizer = optim.RMSprop(model.parameters(), lr=config['lr'])
    loss_fn = nn.CrossEntropyLoss()

    """ Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
    env = gym.make("Pong-v0")
    observation = env.reset()
    
    prev_x = None # used in computing the difference frame
    y_list, LL_list, reward_list = [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0

    start = time.time()

    # Initializing recorders
    update = 0
    loss_tape = []
    our_score_tape = []
    opponent_score_tape = []
    our_score = 0
    opponent_score = 0

    # TRAINING LOOP
    while update < config['max_updates']:

        if config['render']: env.render()

        # preprocess the observation and set input to network to be difference image
        cur_x = utils.preprocess(observation, data_format=config['data_format'])
        if prev_x is None:
            x = np.zeros(cur_x.shape)
        else:
            x = cur_x - prev_x
        prev_x = cur_x

        x_torch = Variable(torch.from_numpy(x).float(), requires_grad=False)
        if config['data_format'] == "array":
            x_torch = x_torch.unsqueeze(dim=0).unsqueeze(dim=0)
        
        if torch.cuda.is_available():
            x_torch = x_torch.cuda(gpu_id)

        # Feedforward through the policy network
        action_prob = model(x_torch)
        
        # Sample an action from the returned probability
        if np.random.uniform() < action_prob.cpu().data.numpy():
            action = 2 # UP
        else:
            action = 3 # DOWN

        # record the log-likelihoods
        y = 1 if action == 2 else 0 # a "fake label"
        NLL = -y*torch.log(action_prob) - (1-y)*torch.log(1 - action_prob)
        LL_list.append(NLL)
        y_list.append(y) # grad that encourages the action that was taken to be taken        TODO: the tensor graph breaks here. Find a way to backpropagate the PG error.

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
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
                if torch.cuda.is_available():
                    Return_i = Return_i.cuda(gpu_id)
                loss = loss + (LL_list[i]*(Return_i)).sum() # .expand_as(LL_list[i])
            loss = loss / len(reward_list)
            print(loss)

            # Backpropagates to compute the gradients
            loss.backward()


            y_list, LL_list, reward_list = [], [], [] # reset array memory

            # Performs parameter update every config['mb_size'] episodes
            if episode_number % config['mb_size'] == 0:

                # Takes one training step
                optimizer.step()
                    
                # Empties the gradients
                optimizer.zero_grad()

                stop = time.time()
                print("PARAMETER UPDATE ------------ {}".format(stop - start))
                start = time.time()

                utils.save_results(save_dir, exp_name, loss_tape, our_score_tape, opponent_score_tape, config)

                update += 1
                if update % 10 == 0: 
                    torch.save(model.state_dict(), os.path.join(save_dir, exp_name, "model_"+model.name()))
                
            # Records the average loss and score of the episode
            loss_tape.append(loss.cpu().data.numpy())

            our_score_tape.append(our_score)
            opponent_score_tape.append(opponent_score)
            our_score = 0
            opponent_score = 0

            # boring book-keeping
            if running_reward is None:
                running_reward = reward_sum
            else:
                running_reward = running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was {0:.2f}. running mean: {1:.2f}'.format(reward_sum, running_reward))
            
            reward_sum = 0
            observation = env.reset() # reset env
            prev_x = None

        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
            if reward == -1:
                opponent_score += 1
                print('ep {0}: game finished, reward: {1:.2f}'.format(episode_number, reward))
            else:
                our_score += 1
                print('ep {0}: game finished, reward: {1:.2f} !!!!!!!!!'.format(episode_number, reward))




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
