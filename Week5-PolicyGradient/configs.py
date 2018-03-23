import collections
import os

class CarefulDict(dict):
    def __init__(self, inp=None):
        # Checks is the input to constructor is already a dict
        if isinstance(inp,dict):
            # If yes, just initializes as usual
            super(CarefulDict,self).__init__(inp)
        else:
            # If not, set each item individually with the overriden __setitem__ method
            super(CarefulDict,self).__init__()
            if isinstance(inp, (collections.Mapping, collections.Iterable)): 
                for k,v in inp:
                    self.__setitem__(k,v)

    def __setitem__(self, k, v):
        try:
            # Looks if the key already exists, if it does, raises an error
            self.__getitem__(k)
            raise ValueError('duplicate key "{0}" found'.format(k))
        
        except KeyError:
            # If the key is unique, just add the tuple to the dict
            super(CarefulDict, self).__setitem__(k, v)


myConfigs = (CarefulDict([

    (0, {  # Imposed hyperparams for Assignment 2, Q1-c (CNN without BatchNorm)
        "data_format": "vector",  # "vector" or "array"
        "input_size": 80*80,
        "output_size": 1,

        "model_type": 'MLP',# CNN or MLP
        "hidden_layers": [200],
        "nonlinearity": "relu",  # "relu", "sigmoid", "tanh"
        "initialization": "glorot",  # "standard", "glorot", "zero", "normal"

        "mb_size": 10,
        "max_updates": 1500,

        "lr": 1e-4,
        "momentum": 0.0,
        "dropout": 0.0, # put 0 for no dropout

        "gamma": 0.99,

        "optimizer": 'adam',

        "is_early_stopping": False,  # True or False
        "L2_hyperparam": 0,  # L2 hyperparameter for a full batch (entire dataset)

        "save_plots": True,
        "resume": False,
        "render": False
        }
    ),

    (1, {  # Imposed hyperparams for Assignment 2, Q1-c (CNN without BatchNorm)
        "data_format": "array",  # "vector" or "array"
        "input_size": 80*80,
        "output_size": 1,

        "model_type": 'CNN',# CNN or MLP
        "nonlinearity": "relu",  # "relu", "sigmoid", "tanh"
        "initialization": "glorot",  # "standard", "glorot", "zero", "normal"

        "mb_size": 10,
        "max_updates": 1500,

        "lr": 1e-4,
        "momentum": 0.0,
        "is_batch_norm": False,

        "gamma": 0.99,

        "optimizer": 'adam',

        "is_early_stopping": False,  # True or False
        "L2_hyperparam": 0,  # L2 hyperparameter for a full batch (entire dataset)

        "save_plots": True,
        "resume": False,
        "render": False
        }
    ),

    (2, {  # MOUNTAIN-CAR
        "input_size": 2,
        "output_size": 3,

        "model_type": 'MLP',# CNN or MLP
        "hidden_layers": [200, 50],
        "nonlinearity": "sigmoid",  # "relu", "sigmoid", "tanh"
        "initialization": "glorot",  # "standard", "glorot", "zero", "normal"

        "mb_size": 10,
        "max_updates": 1500,

        "lr": 1e-4,
        "momentum": 0.0,
        "dropout": 0.0, # put 0 for no dropout

        "gamma": 0.99, # Discount factor
        'epsilon': 0.05, # For epsilon-greedy policy
        'lambda': 0.9, # Eligibility parameter

        "L2_hyperparam": 0,  # L2 hyperparameter for a full batch (entire dataset)

        "use_cuda": False,
        "save_plots": True,
        "resume": False,
        "render": False
        }
    ),




]))