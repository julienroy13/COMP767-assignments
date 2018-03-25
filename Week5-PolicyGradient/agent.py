import os
import numpy as np
import sys
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb

from model import MLP


class REINFORCE:
    def __init__(self, obs_space_size, hidden_sizes, action_space_size, learning_rate, use_cuda, gpu_id):

        self.action_space_size = action_space_size
        self.use_cuda = use_cuda
        self.gpu_id = gpu_id

        # Initializes the policy network and optimizer
        self.policy = MLP(obs_space_size, hidden_sizes, action_space_size, "relu", "standard", verbose=True)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Creates counters
        self.action_count = np.zeros(shape=(self.action_space_size,))

        self.explore_count = 0
        self.exploit_count = 0

        # If GPU is available, sends model on GPU
        if torch.cuda.is_available() and self.use_cuda:
            self.policy.cuda(gpu_id)
            print("USING GPU-{}".format(gpu_id))

        self.policy.train()

    def select_action(self, observation):

        # Transforms the state into a torch Variable
        x = Variable(torch.Tensor([observation]))
        
        if torch.cuda.is_available() and self.use_cuda:
            x = x.cuda(self.gpu_id)
        
        # Forward propagation through policy network
        action_probs = self.policy(x)     
        
        # Samples an action
        action = action_probs.multinomial().data
        
        # Negative log-likelihood of sampled action
        NLL = - torch.log(action_probs[:, action[0,0]]).view(1, -1)

        if int(action) == int(torch.max(action_probs, 1)[1].cpu().data):
            self.exploit_count += 1
        else:
            self.explore_count += 1
        self.action_count[int(action)] += 1

        return int(action), NLL

    def compute_gradients(self, reward_list, NLL_list, gamma):
        
        R = torch.zeros(1, 1)
        loss = 0
        
        # Iterates through the episode in reverse order to compute return for each step
        for i in reversed(range(len(reward_list))):
            
            # Discounts reward
            R = gamma * R + reward_list[i]
            Return_i = Variable(R)
            if torch.cuda.is_available() and self.use_cuda:
                Return_i.cuda(self.gpu_id)
            
            # Loss is the NLL at each step weighted by the return for that step
            loss = loss + (NLL_list[i] * Return_i).squeeze()
        
        # Average to get the total loss
        loss = loss / len(reward_list)
        
        # Backpropagation to compute the gradients
        loss.backward()

        return loss.cpu().data.numpy()

    def update_parameters(self):
            # Clips the gradient and apply the update
            torch.nn.utils.clip_grad_norm(self.policy.parameters(), 40)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def save_policy(self, directory):
        torch.save(self.policy.state_dict(), os.path.join(directory, self.policy.name()+"_ckpt.pkl"))

    def load_policy(self, directory):
        model.load_state_dict(torch.load(os.path.join(directory, "model_" + self.policy.name())))

    def reset_counters(self):

        self.action_count = np.zeros(shape=(self.action_space_size,))

        self.explore_count = 0
        self.exploit_count = 0
