import numpy as np

import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def preprocess(I, data_format):
    """ preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    
    if data_format == "vector":
        I = np.ravel(I)

    return I.astype(np.float)

def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def save_results_PONG(save_dir, exp_name, loss_tape, our_score_tape, opponent_score_tape, config):

    # Creates the folder if necessary
    saving_dir = os.path.join(save_dir, exp_name)
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    # Creates and save the plots
    plt.figure(figsize=(20, 6))

    plt.subplot(1,2,1)
    plt.title("Loss", fontweight='bold')
    plt.plot(loss_tape, color="blue", label="Agent")
    plt.xlabel("Episodes")
    plt.legend(loc='best')

    plt.subplot(1,2,2)
    plt.title("Score", fontweight='bold')
    plt.plot(our_score_tape, color="blue", label="Agent")
    plt.plot(opponent_score_tape, color="orange", label="Opponent")
    plt.ylim(0, 22)
    plt.yticks(np.arange(0, 22, 1))
    plt.xlabel("Episodes")
    plt.legend(loc='best')

    plt.savefig(os.path.join(saving_dir, exp_name + '.png'), bbox_inches='tight')
    plt.close()

    # Save the recording tapes (learning curves) in a file
    log_file = os.path.join(saving_dir, 'log_' + exp_name + '.pkl')
    with open(log_file, 'wb') as f:
        pickle.dump({
            'config': config,
            'loss_tape': loss_tape,
            'our_score_tape': our_score_tape,
            'opponent_score_tape': opponent_score_tape,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    return

def save_results_classicControl(save_dir, exp_name, loss_tape, episode_lengths, config):

    # Creates the folder if necessary
    saving_dir = os.path.join(save_dir, exp_name)
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    # Creates and save the plots
    plt.figure(figsize=(20, 6))

    plt.subplot(1,2,1)
    plt.title("Loss", fontweight='bold')
    plt.plot(loss_tape, color="blue", label="Agent")
    plt.xlabel("Episodes")
    plt.legend(loc='best')

    plt.subplot(1,2,2)
    plt.title("Episode Length", fontweight='bold')
    plt.plot(episode_lengths, color="blue", label="Agent")
    plt.xlabel("Episodes")
    plt.legend(loc='best')

    plt.savefig(os.path.join(saving_dir, exp_name + '.png'), bbox_inches='tight')
    plt.close()

    # Save the recording tapes (learning curves) in a file
    log_file = os.path.join(saving_dir, 'log_' + exp_name + '.pkl')
    with open(log_file, 'wb') as f:
        pickle.dump({
            'config': config,
            'loss_tape': loss_tape,
            'episode_lengths': episode_lengths,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    return

def normalize_observation(obs, env):
    return (obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)

