import gym
import time
import rl_bootcamp
import numpy as np
import ipdb

env = gym.make('RickGrid-v0')

'''
todo:
- action value visualization
- make sure terminal transitions are correct...
- figure out why big reward necessary
'''
# settings
alpha = .05
gamma = 1
epsilon = .1

episodes = 10000
render_episodes = 500  # render once every render_episodes
max_steps = 200  # max steps per episode

# maze environment
# walls = np.array([[0, 1, 0, 0, 0],
#                   [0, 0, 0, 1, 1],
#                   [0, 1, 1, 0, 0],
#                   [0, 1, 0, 0, 0],
#                   [0, 0, 0, 1, 0]], dtype='bool')
walls = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='bool')
# reward_locations = [[2,14,0], [4,19,100]]  # x, y, r
reward_locations = [[4,19,100]]  # x, y, r

rewards = np.full(walls.shape, -1)
is_terminal = np.full(walls.shape, 0, dtype='bool')
for r in reward_locations:
    rewards[r[0], r[1]] = r[2]
    is_terminal[r[0], r[1]] = True
env.__init__(walls=walls, rewards=rewards, is_terminal=is_terminal, start_coords=[4,0])

# empty environment
# sz = (5, 20)
# walls = np.zeros(sz, dtype='bool')
# rewards = np.full(sz, -1)
# rewards[sz[0]-1, sz[1]-1] = 1000
# is_terminal = np.full(sz, 0, dtype='bool')
# is_terminal[sz[0]-1, sz[1]-1] = True
# env.__init__(walls=walls, rewards=rewards, is_terminal=is_terminal)

# initializations
Q = np.zeros((env.observation_space.n, env.action_space.n), dtype='float64')

for i in range(episodes):

    s0 = env.reset()
    is_rendering = (i%render_episodes)==0 and i>0

    for t in range(max_steps):

        # select action
        a = np.argmax(Q[s0])
        if np.random.uniform() < epsilon and not is_rendering:
            a = np.random.randint(0, env.action_space.n)

        # take action, observe next state
        s1, r, done, info = env.step(a)

        # update Q
        target = r if done else r + gamma * np.max(Q[s1])
        Q[s0, a] = Q[s0, a] + alpha * (target - Q[s0, a])
        s0 = s1

        if is_rendering:
            env.render()
            time.sleep(.1)

        if done or t==100:
            # print("Episode finished after {} timesteps".format(t+1))
            break

env.close()