from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
from dm_control import suite
import numpy as np
import matplotlib
import ipdb
import io



class Arm:

    def __init__(self):
        # todo: figure out whether it makes sense to have main and simulation env...
        self.env = suite.load('reacher', 'easy')
        self.env.reset()  # initial randomization appears to require this line... :/
        self.max_steps = self.env._step_limit
        self.Q = np.diag((0,0,1,1,0,0))  # only weight distance from target
        self.R = np.zeros((6,6))         # no control costs for now

    def step(self, action):
        state = self.parse_timestep(self.env.step(action))
        return state

    def rollout(self, actions):
        states = []
        env.reset()
        for i in range(self.max_steps):
            states.append(self.step(action))
        return states

    def cost(self, state, action):
        # J(t) = .5*(x^TQx + uTRt)
        J =  .5 * np.dot(self.state,  np.matmul(self.Q, self.s self tate))
        J += .5 * np.dot(self.action, np.matmul(self.R, self.action))

    def cost_final(self, state):
        J =  .5 * np.dot(self.state,  np.matmul(self.Q, self.state))

    def cost_derivs(self, state, action):
        """ l_x l_u l_xu l_xx l_uu """
        pass

    def state_derivs(self, state, action):
        """ f_x f_u"""
        sim_state = lambda x: self.simulate(x, action)  # keeping action constant
        fx = self.finite_differences(sim_state, state)

        sim_action = lambda x: self.simulate(state, x)  # keeping state constant
        fu = self.finite_differences(sim_action, action)

        return fx, fu

    def simulate(self, state, action):
        """ get subsequent state from (state, action) """
        self.set_state(state)
        state_next = self.step(action)
        return state_next

    def set_state(self, state):
        """ set state without changing target position """
        with self.env.physics.reset_context():
            self.env.physics.data.qpos[:] = state[:2]
            self.env.physics.data.qvel[:] = state[-2:]

    @staticmethod
    def parse_timestep(time_step):
        """ extract state (p0, p1, to_target0, to_target1, v0, v1) """
        return np.concatenate(list(time_step.observation.values()))

    @staticmethod
    def finite_differences(fcn, x, eps=1e-4):
        # todo: vectorize (assuming fcn is vectorized)
        x = np.array(x)
        diffs = []
        for i in range(len(x)):
            x_inc = x.copy()
            x_dec = x.copy()
            x_inc[i] += eps
            x_dec[i] -= eps
            diffs.append((fcn(x_inc) - fcn(x_dec)) / (eps*2))
        return np.array(diffs)

    def render(self):
        return self.env.physics.render(camera_id=0)
