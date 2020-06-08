import time
import numpy as np
import matplotlib.pyplot as plt
import ipdb



class Control:

    def __init__(self, env, Q_init=0):
        self.env = env
        self.Q = np.full((env.observation_space.n, env.action_space.n), Q_init, dtype='float64')
        self.Q_fig = None
        self.Q_init = Q_init

    def select_action(self, s, epsilon):
        a = np.argmax(self.Q[s])
        if np.random.uniform() < epsilon:
            a = np.random.randint(0, self.env.action_space.n)
        return a

    def train(self):
        raise NotImplementedError('"train" method must be defined for Control subclasses')

    def plotQ(self, max_only=False, live_update=False, figsize=None):

        if not self.Q_fig or not live_update:
            self.Q_fig, self.axes = plt.subplots(1, 1 if max_only else 5, figsize=figsize)
            if max_only:
                self.axes = [self.axes]
            self.ims = []
            for i, l in zip(range(5), ['max', 'left', 'right', 'up', 'down']):
                self.ims.append(self.axes[i].imshow(np.zeros(self.env.shape), cmap=plt.get_cmap('hot')))
                self.axes[i].set(xlabel=l, yticks=[], xticks=[])
                if max_only:  # yes, i know this is a hack :(
                    break

        Q_max = np.max(np.reshape(self.Q, (self.env.shape[0], self.env.shape[1], -1)), axis=2)
        lims = [np.min(self.Q), np.max(self.Q)] if not max_only else [np.min(Q_max), np.max(Q_max)]
        Q_max[self.env.walls] = lims[0]   # set the walls equal to the darkest color
        self.ims[0].set_data(Q_max)
        self.ims[0].set_clim(lims[0], lims[1])
        if not max_only:
            for j in range(4):
                im = np.reshape(self.Q[:, j], self.env.shape)
                im[self.env.walls] = lims[0]  # set the walls equal to the darkest color
                self.ims[j + 1].set_data(im)
                self.ims[j + 1].set_clim(lims[0], lims[1])
        plt.pause(.1)

    def resetQ(self):
        self.Q = np.full(self.Q.shape, self.Q_init, dtype='float64')



class QLearning(Control):

    def train(
            self,
            alpha=.05,
            gamma=1,
            epsilon=.1,
            episodes=10000,
            max_steps=100,
            episodes_per_render=None,
            verbose=False,
            show_policy=False):

        steps, rewards = [], []  # number of steps per episode

        for i in range(episodes):

            rewards.append(0)  # keep track of reward per episode
            s0 = self.env.reset()
            is_rendering = episodes_per_render and (i % episodes_per_render) == 0 and i > 0
            if is_rendering:
                self.plotQ()
                self.env.render(Q=self.Q if show_policy else None)
                time.sleep(.1)

            for t in range(max_steps):

                # select action
                a = self.select_action(s0, epsilon if not is_rendering else 0)

                # take action, observe next state
                s1, r, done, info = self.env.step(a)
                rewards[-1] += r

                # update Q
                target = r if done else r + gamma * np.max(self.Q[s1])
                self.Q[s0, a] = self.Q[s0, a] + alpha * (target - self.Q[s0, a])
                s0 = s1

                if is_rendering:
                    self.env.render(Q=self.Q if show_policy else None)
                    time.sleep(.1)

                if done or t==(max_steps-1):
                    steps.append(t+1)
                    if verbose:
                        print('{} steps in episode {}'.format(t+1, i))
                    break

        return steps, rewards



class MonteCarlo(Control):
    '''
    Every visit Monte Carlo control
    '''

    def train(
            self,
            alpha=.05,
            gamma=1,
            epsilon=.1,
            episodes=10000,
            max_steps=100,
            episodes_per_render=None,
            verbose=False,
            show_policy=False,
            live_update=False):

        steps, rewards = [], []  # number of steps per episode
        n = np.zeros(self.Q.shape, dtype='int')  # number of times each state-action pair is encountered

        for i in range(episodes):

            rewards.append(0)  # keep track of reward per episode

            # generate trajectory
            s, a, r = [], [], []
            s.append(self.env.reset())

            is_rendering = episodes_per_render and (i % episodes_per_render) == 0 and i > 0
            if is_rendering:
                self.plotQ(live_update=live_update)
                self.env.render(Q=self.Q if show_policy else None)
                time.sleep(.1)

            for t in range(max_steps):

                # select action
                a.append(self.select_action(s[-1], epsilon if not is_rendering else 0))
                n[s[-1], a[-1]] += 1

                # take action, observe next state
                s_temp, r_temp, done, info = self.env.step(a[-1])
                rewards[-1] += r_temp
                s.append(s_temp)
                r.append(r_temp)

                if is_rendering:
                    self.env.render(Q=self.Q if show_policy else None)
                    time.sleep(.1)

                if done or t==(max_steps-1):
                    steps.append(t+1)
                    s = s[:-1]  # we don't care about the terminal state
                    if verbose:
                        print('{} steps in episode {}'.format(t+1, i))
                    break

            # update returns
            G = 0
            for t in range(len(s)-1, -1, -1):
                G = gamma*G + r[t]

                # non-stationary averaging
                self.Q[s[t],a[t]] = self.Q[s[t],a[t]] + alpha * (G - self.Q[s[t],a[t]])

                # true averaging without bias
                # if self.Q[s[t],a[t]] is not self.Q_init:
                #     self.Q[s[t],a[t]] = self.Q[s[t],a[t]] + (1/n[s[t],a[t]]) * (G - self.Q[s[t],a[t]])  # true average
                # else:
                #     self.Q[s[t], a[t]] = G

                # true averaging with initialization bias
                # self.Q[s[t], a[t]] = self.Q[s[t], a[t]] + (1 / n[s[t], a[t]]) * (G - self.Q[s[t], a[t]])  # true average

        return steps, rewards