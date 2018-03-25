import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

class MLP(nn.Module):

    def __init__(self, inp_size, h_sizes, out_size, nonlinearity, init_type, dropout, verbose=False):

        super(MLP, self).__init__()

        # Hidden layers
        self.hidden = nn.ModuleList([nn.Linear(inp_size, h_sizes[0])])
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))

        # Activation function
        self.nonlinearity = nonlinearity
        if nonlinearity == "relu":
            self.act_fn = F.relu

        elif nonlinearity == "sigmoid":
            self.act_fn = F.sigmoid

        elif nonlinearity == "tanh":
            self.act_fn = F.tanh

        else:
            raise ValueError('Specified activation function "{}" is not recognized.'.format(nonlinearity))

        # Dropout layer
        if dropout != 0:
            self.do_dropout = True
            self.drop = nn.Dropout(p=dropout)
        else:
            self.do_dropout = False

        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

        # Initializes the parameters
        self.init_parameters(init_type)

        self.out_size = out_size

        if verbose:
            print('\nModel Info ------------')
            print(self)
            print("Total number of parameters : {:.2f} M".format(self.get_number_of_params() / 1e6))
            print('---------------------- \n')

    def forward(self, x):

        # Feedforward
        for layer in self.hidden:
            a = layer(x)
            x = self.act_fn(a)
            if self.do_dropout:
                x = self.drop(x)

        if self.out_size == 1:
            output = F.sigmoid(self.out(x)) # Probability of choosing action UP

        else:
            assert len(x.size()) == 1, "Out of the model should be 1D."
            output = F.softmax(self.out(x), dim=0)

        return output

    def init_parameters(self, init_type):

        for module in self.modules():
            if isinstance(module, nn.Linear):

                nn.init.constant(module.bias, 0)

                if init_type == "glorot":
                    nn.init.xavier_normal(module.weight, gain=nn.init.calculate_gain(self.nonlinearity))

                elif init_type == "standard":
                    stdv = 1. / math.sqrt(module.weight.size(1))
                    nn.init.uniform(module.weight, -stdv, stdv)

        for p in self.parameters():
            p.requires_grad = True

    def get_number_of_params(self):

        total_params = 0

        for params in self.parameters():

            total_size = 1
            for size in params.size():
                total_size *= size

            total_params += total_size

        return total_params

    def name(self):
        return "MLP"
