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

    (0, {  # CART-POLE
        "env_name": "CartPole-v1", # Gym environment
        'max_steps': 200,

        "hidden_layers": [128],

        "mb_size": 1,
        "max_episodes": 5000,

        "lr": 1e-3,

        "gamma": 0.99, # Discount factor
        'lambda': 0.9, # Eligibility parameter

        "use_cuda": False,
        'chkp_freq': 100,
        "resume": False,
        "render": False,
        "video_ckpt": [1, 2, 3, 4, 5, 501, 502, 503, 504, 505]
        }
    ),

    (1, {  # MOUNTAIN-CAR
        "env_name": "MountainCar-v0", # Gym environment
        'max_steps': 5000,

        "hidden_layers": [128],

        "mb_size": 1,
        "max_episodes": 5000,

        "lr": 1e-3,

        "gamma": 0.99, # Discount factor
        'lambda': 0.9, # Eligibility parameter

        "use_cuda": False,
        'chkp_freq': 100,
        "resume": False,
        "render": False,
        "video_ckpt": []
        }
    ),

    (2, {  # ACROBOT
        "env_name": "Acrobot-v1", # Gym environment
        'max_steps': 5000,

        "hidden_layers": [128],

        "mb_size": 1,
        "max_episodes": 5000,

        "lr": 1e-3,

        "gamma": 0.99, # Discount factor
        'lambda': 0.9, # Eligibility parameter

        "use_cuda": False,
        'chkp_freq': 100,
        "resume": False,
        "render": False,
        "video_ckpt": [1, 2, 3, 4, 5, 501, 502, 503, 504, 505]
        }
    ),




]))