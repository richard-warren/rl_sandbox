import gym
import numpy as np


mazes = {
    'basic': gym.make('RickGrid-v0',
                      walls=np.array([[0, 0, 1, 0, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 0]], dtype='bool'),
                      terminal_states=[[0, 4, 1]],
                      start_coords=[0, 0],
                      max_steps=100),

    'two_terminal': gym.make('RickGrid-v0',
                             walls=np.array([[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                             [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                             [0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                                             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='bool'),
                             terminal_states=[[4,14,1], [4,19,99]],
                             start_coords=[0,0],
                             max_steps=100),

    'loops': gym.make('RickGrid-v0',
                      walls=np.zeros((7,7), dtype=bool),
                      terminal_states=[[2,2,99]],
                      goodies=[[5,5,5]],
                      start_coords=[0,0],
                      max_steps=100)
}

