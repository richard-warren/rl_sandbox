import time
import numpy as np
import matplotlib.pyplot as plt
import ipdb



class Control:

    def __init__(self, env, Q_init=0):
        self.env = env
        self.Q = np.full((env.observation_space.n, env.action_space.n), Q_init, dtype='float64')
        self.Q_fig = None

    def select_action(self, s, epsilon):
        a = np.argmax(self.Q[s])
        if np.random.uniform() < epsilon:
            a = np.random.randint(0, self.env.action_space.n)
        return a

    def train(self):
        raise NotImplementedError('"train" method must be defined for Control subclasses')

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
            Q_max[self.env.walls] = lims[0]  # set the walls equal to the darkest color
            self.ims[0].set_data(Q_max)
            self.ims[0].set_clim(lims[0], lims[1])
            for j in range(4):
                im = np.reshape(self.Q[:, j], self.env.shape)
                im[self.env.walls] = lims[0]  # set the walls equal to the darkest color
                self.ims[j + 1].set_data(im)
                self.ims[j + 1].set_clim(lims[0], lims[1])
            plt.pause(.1)



class QLearning(Control):

    def train(
            self,
            alpha=.05,
            gamma=1,
            epsilon=.1,
            episodes=10000,
            max_steps=100,
            episodes_per_render=None,
            verbose=True,
            show_policy=False):

        steps, reward = [], []  # number of steps per episode

        for i in range(episodes):

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
        return steps



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
            verbose=True,
            show_policy=False):

        steps, reward = [], []  # number of steps per episode

        for i in range(episodes):

            # generate trajectory
            n = np.zeros(self.Q.shape, dtype='int')  # number of times each state-action pair is encountered
            s, a, r = [], [], []
            s.append(self.env.reset())

            is_rendering = episodes_per_render and (i % episodes_per_render) == 0 and i > 0
            if is_rendering:
                self.plotQ()
                self.env.render(Q=self.Q if show_policy else None)
                time.sleep(.1)

            for t in range(max_steps):

                # select action
                a.append(self.select_action(s[-1], epsilon if not is_rendering else 0))

                # take action, observe next state
                s_temp, r_temp, done, info = self.env.step(a[-1])
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
            for t in range(len(s)-1, 0, -1):
                G = gamma*G + r[t]
                self.Q[s[t],a[t]] = self.Q[s[t],a[t]] + alpha * (G - self.Q[s[t],a[t]])
                # n[s[t],a[t]] = n[s[t],a[t]] + 1
                # self.Q[s[t],a[t]] = self.Q[s[t],a[t]] + (1/n[s[t],a[t]]) * (G - self.Q[s[t],a[t]])  # true average

        return steps