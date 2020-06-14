import numpy as np
import time

# todo: render function with sleep param // how would importance sampling fit into this framework?


class Agent:

    def __init__(self, env, Q_init=0):
        self.env = env
        self.Q_init = Q_init
        self.Q = self.initialize_Q(Q_init)
        self.n = np.zeros(self.Q.shape)  # number of times each state-action pair has been visited

    def select_action(self, state, epsilon):
        if np.random.uniform() < epsilon:
            a = np.random.randint(0, self.env.action_space.n)
        else:
            a = np.argmax(self.Q[state])
        return a

    def step(self, epsilon=0):
        s0 = self.env.state
        a = self.select_action(self.env.state, epsilon)
        s, r, done = self.env.step(a)[:3]
        self.n[s0,a] += 1
        return s, a, r, done

    def rollout(self, max_steps=100, epsilon=0, render=False, online_update=False, alpha=.05, gamma=1, show_policy=True):
        s = self.env.reset()
        states, actions, rewards = [s], [], []
        done = False

        while not done and len(states)<max_steps:
            if not online_update:
                s, a, r, done = self.step(epsilon)
            else:
                s, a, r, done = self.step_update(epsilon=epsilon, alpha=alpha, gamma=gamma)

            [x.append(y) for x, y in zip([states, actions, rewards], [s, a, r])]

            if render:
                self.render(show_policy=show_policy)

        return states, actions, rewards

    def update(self, states, actions, rewards, iterations=1, alpha=.05, gamma=1):
        raise NotImplementedError('update rule is algorithm-specific and should be defined in subclass')

    def initialize_Q(self, Q_init):
        Q = np.full((self.env.observation_space.n, self.env.action_space.n), Q_init, dtype='float64')
        Q[np.reshape([self.env.walls | self.env.is_terminal], self.env.observation_space.n)] = 0  # set values of unreachable states to 0
        return Q

    def render(self, show_policy=False):
        self.env.render(Q=self.Q if show_policy else None)
        time.sleep(.05)


class QLearning(Agent):

    def update(self, states, actions, rewards, iterations=1, alpha=.05, gamma=1):
        for i in range(iterations):
            for s, s_next, a, r in zip(reversed(states[:-1]), reversed(states[1:]), reversed(actions), reversed(rewards)):  # updates start at the end of rollout
                target = r + gamma * np.max(self.Q[s_next])
                self.Q[s,a] = self.Q[s,a] + alpha * (target - self.Q[s,a])

    def step_update(self, epsilon=0, alpha=.05, gamma=1):
        s0 = self.env.state
        s, a, r, done = self.step(epsilon=epsilon)
        self.update([s0, s], [a], [r], alpha=alpha, gamma=gamma)
        return s, a, r, done


class MonteCarlo(Agent):

    def update(self, states, actions, rewards, alpha=.05, gamma=1):
        G = 0
        for s, a, r in zip(reversed(states[:-1]), reversed(actions), reversed(rewards)):
            G = r + gamma * G
            if alpha:
                self.Q[s,a] = self.Q[s,a] + alpha * (G - self.Q[s,a])  # non-stationary average
            else:
                self.Q[s, a] = self.Q[s, a] + (1 / self.n[s, a]) * (G - self.Q[s, a])  # true average with initialization bias


class ExactDP(Agent):

    def value_iteration(self, iterations=1, gamma=1):
        Q_new = self.Q.copy()

        for i in range(iterations):  # todo: randomize state order
            for s in range(self.env.observation_space.n):
                for a in range(self.env.action_space.n):
                    s_next = self.env.P[s,a]
                    a_next = np.argmax(self.Q[s_next])
                    Q_new[s,a] = self.env.R[s,a] + gamma * Q_new[s_next,a_next]  # todo: should really take epsilon into account
