import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
import ipdb

class Agent:

    def __init__(self, env, Q_init=0):
        self.env = env
        self.Q_init = Q_init
        self.Q = self.initialize_q(Q_init)
        self.n = np.zeros(self.Q.shape)  # number of times each state-action pair has been visited

    def select_action(self, state, epsilon):
        if np.random.uniform() < epsilon:
            a = np.random.randint(0, self.env.action_space.n)
        else:
            a = np.random.choice(np.flatnonzero(self.Q[state]==np.max(self.Q[state])))  # break ties randomly
        return a

    def step(self, epsilon=0):
        s0 = self.env.state
        a = self.select_action(self.env.state, epsilon)
        s, r, done = self.env.step(a)[:3]
        self.n[s0,a] += 1
        return s, a, r, done

    def rollout(self, epsilon=0, online_update=False, alpha=.05, gamma=1):
        s = self.env.reset()
        states, actions, rewards = [s], [], []
        done = False

        while not done:
            if not online_update:
                s, a, r, done = self.step(epsilon)
            else:
                s, a, r, done = self.step_update(epsilon=epsilon, alpha=alpha, gamma=gamma)

            [x.append(y) for x, y in zip([states, actions, rewards], [s, a, r])]

        return states, actions, rewards

    def update(self, states, actions, rewards, iterations=1, alpha=.05, gamma=1):
        raise NotImplementedError('update rule is algorithm-specific and should be defined in subclass')

    def initialize_q(self, Q_init):
        Q = np.full((self.env.observation_space.n, self.env.action_space.n), Q_init, dtype='float64')
        Q[np.reshape([self.env.walls | self.env.is_terminal], self.env.observation_space.n)] = 0  # set values of unreachable states to 0
        return Q

    def show_policy(self):
        policy = [self.select_action(s,0) for s in range(self.env.observation_space.n)]
        self.env.render(policy=policy)
        time.sleep(.05)

    def show_q(self):
        im = np.reshape(np.max(self.Q, axis=1), self.env.walls.shape)
        im[self.env.walls] = np.min(im)
        plt.imshow(im, cmap='hot')
        plt.axis('off')
        plt.pause(.01)


class QLearning(Agent):

    def update(self, states, actions, rewards, replays=1, alpha=.05, gamma=1, update_order='forward'):
        '''
        update Q function based on trajectory of states, actions, and rewards
        '''

        for i in range(replays):
            lists = [states[:-1], states[1:], actions, rewards, list(range(len(actions)))]
            if update_order=='forward':
                zipped = zip(*lists)
            elif update_order=='reverse':
                zipped = zip(*[reversed(x) for x in lists])
            elif update_order=='random':
                inds = np.random.choice(list(range(len(actions))), size=len(actions), replace=False)
                zipped = zip(*[np.array(x)[inds] for x in lists])

            for s, s_next, a, r, idx in zipped:
                target = r + gamma * np.max(self.Q[s_next])
                self.Q[s,a] = self.Q[s,a] + alpha * (target - self.Q[s,a])

    def step_update(self, epsilon=0, alpha=.05, gamma=1):
        '''
        take a step while updating Q for online learning
        '''

        s0 = self.env.state
        s, a, r, done = self.step(epsilon=epsilon)
        self.update([s0, s], [a], [r], alpha=alpha, gamma=gamma)
        return s, a, r, done

    def train(self, iterations=1000, epsilon=0, online_update=True, alpha=.05, gamma=1, replays=1, update_order='forward'):
        '''
        train agent
        '''

        steps, reward = [], []  # number of steps and total reward for each iterations
        for i in range(iterations):
            if online_update:
                r = self.rollout(epsilon=epsilon, online_update=True, alpha=alpha, gamma=gamma)[2]
            else:
                s, a, r = self.rollout(epsilon=epsilon, online_update=False)
                self.update(s, a, r, replays=replays, update_order=update_order)
            steps.append(len(r))
            reward.append(sum(r))
        return steps, reward


class MonteCarlo(Agent):

    def update(self, states, actions, rewards, alpha=None, gamma=1):
        '''
        update Q function based on trajectory of states, actions, and rewards
        '''

        G = 0
        for s, a, r in zip(reversed(states[:-1]), reversed(actions), reversed(rewards)):
            G = r + gamma * G
            if alpha:
                self.Q[s,a] = self.Q[s,a] + alpha * (G - self.Q[s,a])  # non-stationary average
            else:
                self.Q[s, a] = self.Q[s, a] + (1 / self.n[s, a]) * (G - self.Q[s, a])  # true average with initialization bia

    def train(self, iterations=1000, epsilon=0, alpha=None, gamma=1):
        '''
        train agent
        '''

        steps, reward = [], []  # number of steps and total reward for each iterations
        for i in range(iterations):
            s, a, r = self.rollout(epsilon=epsilon, online_update=False, gamma=1)
            self.update(s, a, r, alpha=alpha, gamma=gamma)
            steps.append(len(r))
            reward.append(sum(r))
        return steps, reward


class DP_approx(Agent):

    def evaluate_and_update(self, eval_iterations=10, alpha=.05, gamma=1, epsilon=.05):
        '''
        evaluates the policy defined by self.Q
        '''

        Q_new = self.Q.copy()
        for i in range(eval_iterations):
            for s, a in itertools.product(range(self.Q.shape[0]), range(self.Q.shape[1])):
                r = self.env.R[s,a]
                s_next = self.env.P[s,a]
                a_next = self.select_action(s_next, epsilon)
                target = r + gamma * Q_new[s_next, a_next]
                Q_new[s,a] = Q_new[s,a] + alpha * (target - Q_new[s,a])
        self.Q = Q_new


class DP_exact(Agent):

    def __init__(self, env):
        super().__init__(env)
        self.V = np.zeros((self.env.observation_space.n, self.env.max_steps + 1))         # (state X time) optimal value of each state at each time
        self.A = np.zeros((self.env.observation_space.n, self.env.max_steps), dtype=int)  # (state X time) best actions to take in each state at each time

    def select_action(self, s, t):
        return self.A[s,t]

    def step(self, epsilon=None):
        s0 = self.env.state
        a = self.select_action(self.env.state, self.env.time)
        s, r, done = self.env.step(a)[:3]
        self.n[s0,a] += 1
        return s, a, r, done

    def solve(self):
        for i in reversed(range(self.env.max_steps)):
            for s in range(self.env.observation_space.n):
                v_next = self.env.R[s] + self.V[self.env.P[s], i + 1]  # R is the reward dynamics; S is the state dynamics
                self.A[s, i] = np.argmax(v_next)
                self.V[s, i] = np.max(v_next)

    def show_policy(self, t=0):
        policy = [self.select_action(s,t) for s in range(self.env.observation_space.n)]  # show the action selected at time 0
        self.env.render(policy=policy)
        time.sleep(.05)



