import gym
import numpy as np
import ipdb


class RickGrid(gym.Env):

    def __init__(self,
                 walls=np.zeros((10,10), dtype='bool'),
                 rewards=[[9,9,10]],  # [column, row, reward]
                 nonterminal_reward=-1,
                 start_coords=[0,0],
                 random_start=False
                 ):

        self.walls = walls
        self.shape = walls.shape
        self.action_space = gym.spaces.Discrete(4)  # left, right, up, down
        self.observation_space = gym.spaces.Discrete(walls.size)
        self.start_state = self.coords_to_state(start_coords)
        self.bump_location = None  # location of wall when agent is trying to walk into a wall
        self.random_start = random_start

        # construct reward and is_terminal matrices
        self.rewards = np.full(np.shape(walls), nonterminal_reward)
        self.is_terminal = np.zeros(np.shape(walls), dtype='bool')
        for i in range(len(rewards)):
            self.rewards[rewards[i][0], rewards[i][1]] = rewards[i][2]
            self.is_terminal[rewards[i][0], rewards[i][1]] = True

        # get start state
        if random_start:
            self.state = self.get_random_state()
        else:
            self.state = self.start_state

    def step(self, action):

        # update state
        deltas = [(0,-1), (0,1), (-1,0), (1,0)]  # row, col change for left, right, up, down
        delta = deltas[action]
        self.bump_location = None

        coords = self.state_to_coords(self.state)
        coords[0] += delta[0]
        coords[1] += delta[1]

        # if action moves out of the grid, return to previous position
        if coords[0]<0 or coords[0]>=self.walls.shape[0] or coords[1]<0 or coords[1]>=self.walls.shape[1]:
            self.bump_location = coords
            coords = self.state_to_coords(self.state)

        # if action moves into a wall, return to previous position
        if self.walls[coords[0], coords[1]]:
            self.bump_location = coords
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


    def get_random_state(self):
        # ipdb.set_trace()
        valid_inds = np.where(~(self.is_terminal | self.walls))
        i = np.random.randint(len(valid_inds[0]))
        state = self.coords_to_state([valid_inds[0][i], valid_inds[1][i]])
        return state



    def render(self, mode='human'):

        # shift all coordinates +1 because graphics will include walls surrounding maze
        coords = self.state_to_coords(self.state)
        coords = [coords[0]+1, coords[1]+1]
        bump = [self.bump_location[0]+1, self.bump_location[1]+1] if self.bump_location else None
        walls = np.ones((self.shape[0]+2, self.shape[1]+2), dtype='bool')
        walls[1:-1,1:-1] = self.walls
        is_terminal = np.zeros((self.shape[0] + 2, self.shape[1] + 2), dtype='bool')
        is_terminal[1:-1,1:-1] = self.is_terminal

        # construct string representation of maze
        maze_str = []
        for r in range(self.shape[0]+2):
            maze_str.append('\n')
            for c in range(self.shape[1]+2):
                if bump and bump[0]==r and bump[1]==c:
                    symbol = 'XX'
                elif coords[0]==r and coords[1]==c:
                    symbol = '☺ '
                elif walls[r,c]:
                    symbol = '██'
                elif is_terminal[r,c]:
                    symbol = '{:2d}'.format(self.rewards[r-1,c-1])
                else:
                    # symbol = '{:2d}'.format(self.rewards[r-1,c-1])  # uncomment to show non-terminal values in maze
                    symbol = '  '
                maze_str[r] += (symbol)
        print(''.join(maze_str))

        return maze_str


    def reset(self):
        self.state = self.get_random_state() if self.random_start else self.start_state
        return self.state





