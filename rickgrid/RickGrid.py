import gym
import numpy as np
import platform


class RickGrid(gym.Env):

    def __init__(self,
                 walls=np.zeros((10,10), dtype='bool'),
                 terminal_states=[[9, 9, 10]],  # [column, row, reward]
                 nonterminal_reward=-1,
                 start_coords=[0,0],
                 random_start=False,
                 goodies=None,
                 max_steps=100
                 ):

        self.walls = walls
        self.shape = walls.shape
        self.action_space = gym.spaces.Discrete(4)  # left, right, up, down
        self.observation_space = gym.spaces.Discrete(walls.size)
        self.start_state = self.coords_to_state(start_coords)
        self.random_start = random_start
        self.max_steps = max_steps
        self.time = 0

        # construct reward and is_terminal matrices
        self.rewards = np.full(np.shape(walls), nonterminal_reward)
        self.is_terminal = np.zeros(np.shape(walls), dtype='bool')
        self.is_goody = np.zeros(np.shape(walls), dtype='bool')
        for i in range(len(terminal_states)):
            self.rewards[terminal_states[i][0], terminal_states[i][1]] = terminal_states[i][2]
            self.is_terminal[terminal_states[i][0], terminal_states[i][1]] = True
        if goodies:
            for i in range(len(goodies)):
                self.rewards[goodies[i][0], goodies[i][1]] = goodies[i][2]
                self.is_goody[goodies[i][0], goodies[i][1]] = True

        # get start state
        if random_start:
            self.state = self.get_random_state()
        else:
            self.state = self.start_state

        # make transition matrices
        self.P, self.R, self.D, = self.make_transition_matrices()

    def make_transition_matrices(self):

        deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # row, col change for left, right, up, down
        size = (self.observation_space.n, self.action_space.n)
        P, R = np.zeros(size, dtype=int), np.zeros(size, dtype=int)
        D = np.zeros(size, dtype=bool)

        for s in range(self.observation_space.n):
            for a in range(self.action_space.n):

                loc = self.state_to_coords(s)

                # terminal state treated as infinite loop with zero reward
                if self.is_terminal[loc[0], loc[1]]:
                    P[s,a] = s
                    R[s,a] = 0
                    D[s,a] = True

                else:
                    # update state
                    loc[0] += deltas[a][0]
                    loc[1] += deltas[a][1]

                    # if action moves out of the grid or into a wall the agent doesn't move
                    x_invalid = loc[0] < 0 or loc[0] >= self.walls.shape[0]
                    y_invalid = loc[1] < 0 or loc[1] >= self.walls.shape[1]

                    if x_invalid or y_invalid or self.walls[loc[0], loc[1]]:
                        P[s,a] = s
                        loc = self.state_to_coords(s)
                    else:
                        P[s,a] = self.coords_to_state(loc)

                    # determine reward
                    R[s,a] = self.rewards[loc[0], loc[1]]

                    # determine whether in terminal state
                    D[s,a] = self.is_terminal[loc[0], loc[1]]

        return P, R, D

    def step(self, action):
        s0 = self.state
        self.state = self.P[s0, action]
        reward = self.R[s0, action]
        self.time+=1
        done = self.D[s0, action] or self.time >= self.max_steps
        return self.state, reward, done, {}

    def state_to_coords(self, state):
        loc = list(np.unravel_index(state, self.shape))
        return loc

    def coords_to_state(self, coords):
        state = np.ravel_multi_index(coords, self.shape)
        return state

    def get_random_state(self):
        valid_inds = np.where(~(self.is_terminal | self.walls))
        i = np.random.randint(len(valid_inds[0]))
        state = self.coords_to_state([valid_inds[0][i], valid_inds[1][i]])
        return state

    def render(self, mode='human', policy=None):
        # shift all coordinates +1 because graphics will include walls surrounding maze
        loc = self.state_to_coords(self.state)
        loc = [loc[0]+1, loc[1]+1]
        walls = np.ones((self.shape[0]+2, self.shape[1]+2), dtype='bool')
        walls[1:-1,1:-1] = self.walls
        is_terminal = np.zeros((self.shape[0] + 2, self.shape[1] + 2), dtype='bool')
        is_terminal[1:-1,1:-1] = self.is_terminal
        is_goody = np.zeros((self.shape[0] + 2, self.shape[1] + 2), dtype='bool')
        is_goody[1:-1, 1:-1] = self.is_goody

        # construct string representation of maze
        maze_str = []
        for r in range(self.shape[0]+2):
            maze_str.append('\n')
            for c in range(self.shape[1]+2):
                if loc[0]==r and loc[1]==c:
                    symbol = '☺ '
                elif walls[r,c]:
                    symbol = '██'
                elif is_terminal[r,c] or is_goody[r,c]:
                    symbol = '{:2d}'.format(self.rewards[r-1,c-1])
                else:
                    if policy is None:
                        symbol = '  '
                    else:
                        s = self.coords_to_state((r-1,c-1))
                        symbol = [' ◃', ' ▹', '▵ ', '▿ '][policy[s]]
                        # symbol = [' <', ' >', '^ ', 'v '][policy[s]]
                maze_str[r] += (symbol)
        print(''.join(maze_str))

    def reset(self, start_loc=None):
        if not start_loc:
            self.state = self.get_random_state() if self.random_start else self.start_state
        else:
            self.state = self.coords_to_state(start_loc)
        self.time = 0
        return self.state
