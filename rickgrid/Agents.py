import numpy as np
import time
import ipdb

# todo: render function with sleep param // how would importance sampling fit into this framework?


class Agent:

    def __init__(self, env, Q_init=0):
        self.env = env
        self.Q_init = Q_init
        self.Q = self.set_Q(Q_init)

    def select_action(self, state, epsilon):
        if np.random.uniform() < epsilon:
            a = np.random.randint(0, self.env.action_space.n)
        else:
            a = np.argmax(self.Q[state])
        return a

    def step(self, epsilon=0):
        a = self.select_action(self.env.state, epsilon)
        s, r, done = self.env.step(a)[:3]
        return s, a, r, done

    def rollout(self, render=False, max_steps=100, epsilon=0):
        s = self.env.reset()
        states, actions, rewards = [s], [], []
        done = False

        while not done and len(states)<max_steps:
            s, a, r, done = self.step(epsilon)
            [x.append(y) for x, y in zip([states, actions, rewards], [s, a, r])]
            if render:
                self.env.render(Q=self.Q)
                time.sleep(.05)

        return states, actions, rewards

    def update(self, states, actions, rewards, iterations=1, alpha=.05, gamma=1):
        raise NotImplementedError('Update rule is algorithm-specific and should be defined in subclass')

    def set_Q(self, Q_init):
        Q = np.full((self.env.observation_space.n, self.env.action_space.n), Q_init, dtype='float64')
        Q[np.reshape([self.env.walls | self.env.is_terminal], self.env.observation_space.n)] = 0  # set values of unreachable states to 0
        return Q


class QLearning(Agent):

    def update(self, states, actions, rewards, iterations=1, alpha=.05, gamma=1):
        for i in range(iterations):
            # for s, s_next, a, r in zip(states[:-1], states[1:], actions, rewards):
            for s, s_next, a, r in zip(reversed(states[:-1]), reversed(states[1:]), reversed(actions), reversed(rewards)):  # todo:
                target = r + gamma * np.max(self.Q[s_next])
                self.Q[s,a] = self.Q[s,a] + alpha * (target - self.Q[s,a])

    # def rollout_update(self, render=False, max_steps=100, epsilon=0, alpha=.05, gamma=1):
    #     s = self.env.reset()
    #     states, actions, rewards = [s], [], []
    #     done = False
    #
    #     while not done and len(states) < max_steps:
    #         s, a, r, done = self.step(epsilon)
    #         [x.append(y) for x, y in zip([states, actions, rewards], [s, a, r])]
    #         if render:
    #             self.env.render(Q=self.Q)
    #             time.sleep(.05)
    #
    #         target = r + gamma * np.max(self.Q[states[-1]])
    #         self.Q[states[-2], a] = self.Q[states[-2], a] + alpha * (target - self.Q[states[-2], a])
    #
    #     return states[:-1], actions, rewards


class MonteCarlo(Agent):

    def update(self, states, actions, rewards, alpha=.05, gamma=1):

        G = 0
        for s, a, r in zip(states, actions, rewards):
            G = r + gamma * G

            # non-starionary averaging
            self.Q[s,a] = self.Q[s,a] + alpha * (G - self.Q[s,a])

            # true averaging without bias
            # if self.Q[s,a] is not self.Q_init:  # this is a hack // should keep track of states that have and have not been updated
            #     self.Q[s,a] = self.Q[s,a] + (1/self.n[s,a]) * (G - self.Q[s,a])  # true average
            # else:
            #     self.Q[s,a] = G

            # true averaging with initialization bias
            # self.Q[s,a] = self.Q[s,a] + (1/self.n[s,a]) * (G - self.Q[s,a])  # true average