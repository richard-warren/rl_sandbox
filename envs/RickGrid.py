import gym
import numpy as np
import ipdb


class RickGrid(gym.Env):

    def __init__(self,
                 walls = np.zeros((3,3), dtype='bool'),
                 rewards = np.array(([-1,-1,0], [-1,-1,-1], [-1,-1,-1])),
                 is_terminal = np.array([[0,0,1], [0,0,0], [0,0,0]], dtype='bool'),
                 start_coords = [0,0]
                 ):

        self.walls = walls
        self.rewards = rewards
        self.is_terminal = is_terminal
        self.shape = walls.shape
        self.action_space = gym.spaces.Discrete(4)  # left, right, up, down
        self.observation_space = gym.spaces.Discrete(walls.size)
        self.state = self.coords_to_state(start_coords)
        self.start_state = self.state
        # todo: initialize state // check that all matrices are the same size


    def step(self, action):

        # update state
        deltas = [(0,-1), (0,1), (-1,0), (1,0)]  # row, col change for left, right, up, down
        delta = deltas[action]

        coords = self.state_to_coords(self.state)
        coords[0] += delta[0]
        coords[1] += delta[1]

        # if action moves out of the grid, return to previous position
        if coords[0]<0 or coords[0]>=self.walls.shape[0] or coords[1]<0 or coords[1]>=self.walls.shape[1]:
            coords = self.state_to_coords(self.state)

        # if action moves into a wall, return to previous position
        if self.walls[coords[0], coords[1]]:
            coords = self.state_to_coords(self.state)

        # determine reward
        reward = self.rewards[coords[0], coords[1]]

        # determine whether in terminal state
        done = self.is_terminal[coords[0], coords[1]]

        # update state
        self.state = self.coords_to_state(coords)

        return self.state, reward, done, {}


    def state_to_coords(self, state):
        coords = list(np.unravel_index(state, self.shape))
        return coords

    def coords_to_state(self, coords):
        state = np.ravel_multi_index(coords, self.shape)
        return state


    def render(self, mode='human'):

        coords = self.state_to_coords(self.state)

        print('\n' + '╔' + '══'*(self.shape[1]) + '═╗')
        for r in range(self.shape[0]):
            print('║ ', end='')
            for c in range(self.shape[1]):
                if coords[0]==r and coords[1]==c:
                    symbol = '██'
                elif self.walls[r,c]:
                    symbol = 'XX'
                elif self.is_terminal[r,c]:
                    symbol = 'O '
                else:
                    symbol = '  '
                print(symbol, end='')
            print('║')
        print('╚' + '══'*(self.shape[1]) + '═╝')

    def reset(self):
        self.state = self.start_state
        return self.state





