import gym
import rickgrid
from rickgrid.mazes import mazes
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from rickgrid import Control


# settings
# loops = 3
episodes_per_loop = 100000
env = gym.make('RickGrid-v0', **mazes[0], random_start=True)

# ctl = Control.MonteCarlo(env, Q_init=0)
# ctl.train(alpha=.1, gamma=1, epsilon=.2,
#           episodes=episodes_per_loop,
#           episodes_per_render=1000,
#           verbose=False,
#           show_policy=True)

ctl = Control.QLearning(env, Q_init=0)
ctl.train(alpha=.1, gamma=1, epsilon=.1,
          episodes=episodes_per_loop,
          episodes_per_render=1000,
          verbose=False)

# steps = np.zeros((loops, episodes_per_loop), dtype=int)
# for l in range(loops):
#     ctl = Control.QLearning(env)
#     steps[l] = ctl.train(alpha=.05, gamma=1, epsilon=.1,
#                          episodes=episodes_per_loop,
#                          episodes_per_render=200,
#                          verbose=False)
#
# smoothing = 21
# smoothed = np.apply_along_axis(lambda x: scipy.signal.medfilt(x, smoothing), 1, steps)
#
# # plt.close('all')
# mean = np.mean(smoothed, 0)
# std = np.std(smoothed, 0)
# plt.plot(mean)
# plt.fill_between(np.arange(episodes_per_loop), mean+std, mean-std, alpha=.4)