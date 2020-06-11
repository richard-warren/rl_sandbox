import gym
import rickgrid
from rickgrid.mazes import mazes
import numpy as np
import matplotlib.pyplot as plt
from rickgrid import Agents
import ipdb

# make environment
# env = gym.make('RickGrid-v0',
#                walls = np.array([[0,0,1,0,0],
#                                  [0,0,1,0,0],
#                                  [0,0,1,0,0],
#                                  [0,0,1,0,0],
#                                  [0,0,0,0,0]], dtype='bool'),
#                terminal_states = [[0,4,1]],     # each terminal_state characterized by [row, col, value]
#                start_coords = [0,0],    # where the agent starts
#                nonterminal_reward = -1)   # -1 per non-terminal step
env = gym.make('RickGrid-v0', **mazes[0], random_start=False)

# make agent
agent = Agents.QLearning(env, Q_init=10)
# agent = Agents.MonteCarlo(env)


im = plt.imshow(np.reshape(np.max(agent.Q,1), env.walls.shape), cmap=plt.get_cmap('hot'))
plt.pause(.1)

for i in range(10000):
    is_rendering = (i % 1000) == 0 and i > 0

    states, actions, rewards = agent.rollout(render=False, epsilon=.5)
    agent.update(states, actions, rewards, iterations=5, alpha=.1)
    # agent.rollout_update(render=is_rendering, max_steps=100, epsilon=.1, alpha=.5)

    if is_rendering:
        im.set_data(np.reshape(np.max(agent.Q,1), env.walls.shape))
        im.set_clim(np.min(agent.Q), np.max(agent.Q))
        plt.pause(.1)
        env.render(Q=agent.Q)

print('Q max {:.2f}, Q min {:.2f}'.format(np.max(agent.Q), np.min(agent.Q)))
# ipdb.set_trace()
