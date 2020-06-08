import gym
import rickgrid
from rickgrid.mazes import mazes
import numpy as np
import matplotlib.pyplot as plt
from rickgrid import Control

# define simple maze
env = gym.make('RickGrid-v0',
               walls = np.array([[0,0,1,0,0],
                                 [0,0,1,0,0],
                                 [0,0,1,0,0],
                                 [0,0,1,0,0],
                                 [0,0,0,0,0]], dtype='bool'),
               rewards = [[0,4,1]],     # each reward is a terminal state characterized by [row, col, value]
               start_coords = [0,0],    # where the agent starts
               nonterminal_reward=-1)   # -1 per non-terminal step



# set up Monte Carlo object
ctl_mc = Control.MonteCarlo(env)

# train, recording number of steps per episode
env.random_start = True
steps_mc, rewards_mc = ctl_mc.train(alpha=.05, gamma=1, epsilon=.1,
                                    episodes=100000,
                                    episodes_per_render=500,
                                    live_update=True,
                                    show_policy=True)
