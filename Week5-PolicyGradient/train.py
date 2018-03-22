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

    while True:
        if config['render']: env.render()

        # preprocess the observation and set input to network to be difference image
        cur_x = utils.preprocess(observation)
        if prev_x is None:
            x = np.zeros(config['input_size'])
        else:
            x = cur_x - prev_x
        prev_x = cur_x

        x_torch = Variable(torch.from_numpy(x).float())

        # Feedforward through the policy network
        action_prob = model(x_torch)
        
        # Sample an action from the returned probability
        if np.random.uniform() < action_prob.data.numpy():
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

        if done: # an episode finished
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            episode_LL = torch.stack(LL_list)
            episode_y = Variable(torch.unsqueeze(torch.FloatTensor(y_list), dim=1))
            
            episode_rewards = np.array(reward_list)
            y_list, LL_list, reward_list = [], [], [] # reset array memory

            # compute the discounted reward backwards through time
            episode_returns = utils.discount_rewards(episode_rewards, config['gamma'])
            
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            episode_returns -= np.mean(episode_returns)
            episode_returns /= np.std(episode_returns)

            # send returns to torch
            episode_returns = Variable(torch.unsqueeze(torch.from_numpy(episode_returns), dim=1).float())

            # Computes the loss
            print("episode_LL.size() : {}".format(episode_LL.size()))
            print("episode_y.size() : {}".format(episode_y.size()))
            print("episode_returns.size() : {}".format(episode_returns.size()))

            print("episode_LL.type() : {}".format(episode_LL.data.type()))
            print("episode_y.data.type() : {}".format(episode_y.data.type()))
            print("episode_returns.type() : {}".format(episode_returns.data.type()))
            
            loss = torch.mean(episode_LL * episode_returns) # modulate the gradient with advantage (PG magic happens right here.)
            print(loss)

            # Backpropagates to compute the gradients
            loss.backward()
            print("HERE")

            # Performs parameter update every config['mb_size'] episodes
            if episode_number % config['mb_size'] == 0:
                
                # Takes one training step
                optimizer.step()
                    
                # Empties the gradients
                optimizer.zero_grad()

            # boring book-keeping
            if running_reward is None:
                running_reward = reward_sum
            else:
                running_reward = running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was {0:.2f}. running mean: {1:.2f}'.format(reward_sum, running_reward))
            
            if episode_number % 100 == 0: 
                torch.save(model.state_dict(), os.path.join(save_dir, exp_name, "model"))
            
            reward_sum = 0
            observation = env.reset() # reset env
            prev_x = None

            if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                if reward == -1:
                    print('ep {0}: game finished, reward: {1:.2f}'.format(episode_number, reward))
                else:
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
