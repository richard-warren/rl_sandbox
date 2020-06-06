import time
import numpy as np
import matplotlib.pyplot as plt


class QLearning:

    def __init__(self, env, Q_init=0):
        self.env = env
        self.Q = np.full((env.observation_space.n, env.action_space.n), Q_init, dtype='float64')
        self.Q_fig = None

    def train(
            self,
            alpha=.05,
            gamma=1,
            epsilon=.1,
            episodes=10000,
            max_steps=100,
            episodes_per_render=None,
            verbose=True):

        steps = []  # number of steps per episode

        # initialize value function plots
        if episodes_per_render:
            self.plotQ()

        for i in range(episodes):

            s0 = self.env.reset()
            is_rendering = episodes_per_render and (i % episodes_per_render) == 0 and i > 0
            if is_rendering:
                self.plotQ()

            for t in range(max_steps):

                # select action
                a = np.argmax(self.Q[s0])
                if np.random.uniform() < epsilon and not is_rendering:
                    a = np.random.randint(0, self.env.action_space.n)

                # take action, observe next state
                s1, r, done, info = self.env.step(a)

                # update Q
                target = r if done else r + gamma * np.max(self.Q[s1])
                self.Q[s0, a] = self.Q[s0, a] + alpha * (target - self.Q[s0, a])
                s0 = s1

                if is_rendering:
                    self.env.render()
                    time.sleep(.1)

                if done or t==(max_steps-1):
                    steps.append(t+1)
                    if verbose:
                        print('{} steps in episode {}'.format(t+1, i))
                    break
        return steps

    def plotQ(self):

        if not self.Q_fig:
            self.Q_fig, self.axes = plt.subplots(5, 1, figsize=(4, 5))
            self.ims = []
            for i, l in zip(range(5), ['max', 'left', 'right', 'up', 'down']):
                self.ims.append(self.axes[i].imshow(np.zeros(self.env.shape), cmap=plt.get_cmap('hot')))
                self.axes[i].set(ylabel=l, yticks=[], xticks=[])
        else:
            Q_max = np.max(np.reshape(self.Q, (self.env.shape[0], self.env.shape[1], -1)), axis=2)
            lims = [np.min(self.Q), np.max(self.Q)]
            self.ims[0].set_data(Q_max)
            self.ims[0].set_clim(lims[0], lims[1])
            for j in range(4):
                self.ims[j + 1].set_data(np.reshape(self.Q[:, j], self.env.shape))
                self.ims[j + 1].set_clim(lims[0], lims[1])
            plt.pause(.1)







