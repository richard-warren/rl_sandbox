import gym
import time
import rickgrid
from rickgrid.mazes import mazes
import numpy as np
import matplotlib.pyplot as plt
import ipdb


# settings
alpha = .05
gamma = 1
epsilon = .1

episodes = 100000
render_episodes = 500  # render once every render_episodes
max_steps = 100  # max steps per episode


# make environment
env = gym.make('RickGrid-v0', **mazes[0], random_start=True)

# plot value function
fig, axes = plt.subplots(5,1,figsize=(4,5))
ims = []
for i, l in zip(range(5), ['max', 'left', 'right', 'up', 'down']):
    ims.append(axes[i].imshow(np.zeros(env.shape), cmap=plt.get_cmap('hot')))
    axes[i].set(ylabel=l, yticks=[], xticks=[])

# initializations
Q = np.zeros((env.observation_space.n, env.action_space.n), dtype='float64')

for i in range(episodes):

    s0 = env.reset()
    is_rendering = (i%render_episodes)==0 and i>0
    if is_rendering:
        Q_max = np.max(np.reshape(Q, (env.shape[0], env.shape[1], -1)), axis=2)
        lims = [np.min(Q), np.max(Q)]
        ims[0].set_data(Q_max)
        ims[0].set_clim(lims[0], lims[1])
        for j in range(4):
            ims[j+1].set_data(np.reshape(Q[:,j], env.shape))
            ims[j+1].set_clim(lims[0], lims[1])
        plt.pause(.1)


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