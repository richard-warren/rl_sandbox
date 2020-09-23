from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
# from dm_control import suite
import numpy as np
import matplotlib
import ipdb
import io



class Env:

    def __init__(self):
        pass

    def simulate(self, state, action):
        """ get subsequent state from (state, action) """
        self.set_state(state)
        state_next = self.step(action)
        return state_next

    def rollout(self, actions):
        # todo: check that doesn't exceed max_steps
        states, costs, costs_derivs = [], [], []
        states.append(self.reset())

        for action in actions:
            cost, cost_derivs = self.cost(states[-1], action)
            costs.append(cost)
            costs_derivs.append(cost_derivs)
            states.append(self.step(action))

        cost, cost_derivs = self.cost_final(states[-1])
        costs.append(cost)
        costs_derivs.append(cost_derivs)

        return states, costs, costs_derivs

    def state_derivs(self, state, action):  # meta-class
        """ f_x, f_u """
        state = np.array(state, dtype='float64')
        action = np.array(action, dtype='float64')

        sim_state = lambda x: self.simulate(x, action)  # keeping action constant
        f_x = self.finite_differences(sim_state, state)

        sim_action = lambda x: self.simulate(state, x)  # keeping state constant
        f_u = self.finite_differences(sim_action, action)

        return dict(f_x=f_x, f_u=f_u)

    @staticmethod
    def finite_differences(fcn, x, eps=1e-4):  # meta-class
        # todo: vectorize (assuming fcn is vectorized)
        x = np.array(x)
        diffs = []
        for i in range(len(x)):
            x_inc = x.copy()
            x_dec = x.copy()
            x_inc[i] += eps
            x_dec[i] -= eps
            diffs.append((fcn(x_inc) - fcn(x_dec)) / (eps*2))
        return np.array(diffs).T


class PointMass(Env):

    def __init__(self, dt=.05, arena_size=(1,1), mass=1, max_time=10):
        self.mass = mass
        self.dt = dt
        self.xlim = (-arena_size[0]/2, arena_size[0]/2)
        self.ylim = (-arena_size[1]/2, arena_size[1]/2)
        self.max_steps = int(max_time // dt)
        self.reset()  # set initial target and point positions

        # graphic objects
        self.fig = plt.figure(figsize=(2.5,2.5))
        self.ax = plt.axes(xlim=(-arena_size[0]*.6, arena_size[0]*.6),
                           ylim=(-arena_size[1]*.6, arena_size[1]*.6))
        plt.axis('off')
        self.plt_circle = plt.plot(self.state[0], self.state[1], marker='o', ms=10)[0]
        self.plt_target = plt.plot(self.target[0],   self.target[1],   marker='o', ms=10, alpha=.5)[0]
        sz = arena_size
        plt.plot([-sz[0]/2, sz[0]/2, sz[0]/2, -sz[0]/2, -sz[0]/2],
                 [sz[1]/2, sz[1]/2, -sz[1]/2, -sz[1]/2, sz[1]/2],
                 color='black', linewidth=4)  # arena walls
        plt.close()

    def reset(self, reset_target=True):
        self.t = 0
        self.state = np.array([0,0,0,0], dtype='float64')  # (pos_x, pos_y, vel_x, vel_y)
        if reset_target:
            self.target = np.random.uniform((self.xlim[0], self.ylim[0]),
                                            (self.xlim[1], self.ylim[1]))
        return self.state.copy()

    def step(self, action):
        self.state[2:] += np.array(action, dtype='float64') * self.dt
        self.state[:2] += self.state[2:] * self.dt
        self.t += self.dt
        return self.state.copy()

    def cost(self, state, action):
        cost = 0.5 * ((self.state[:2] - self.target)**2).sum()
        derivs = {
            'l_x': np.concatenate(((self.state[:2]-self.target), [0,0])),
            'l_u': np.zeros(2),
            'l_ux': np.zeros((2,4)),
            'l_xx': np.diag((1,1,0,0)),
            'l_uu': np.zeros((2,2))
        }
        return cost, derivs

    def cost_final(self, state):
        cost = 0.5 * ((self.state[:2] - self.target)**2).sum()
        derivs = {
            'l_x': np.concatenate(((self.state[:2]-self.target), [0,0])),
            'l_u': np.zeros(2),
            'l_ux': np.zeros((2,4)),
            'l_xx': np.diag((1,1,0,0)),
            'l_uu': np.zeros((2,2))
        }
        return cost, derivs

    def set_state(self, state):
        """ set state without changing target position """
        self.state = state.copy()

    def render(self, dpi=200, show_plot=False):
        pos, vel = self.state[:2], self.state[2:]

        # update graphics
        self.plt_circle.set_xdata(self.state[0])
        self.plt_circle.set_ydata(self.state[1])
        self.plt_target.set_xdata(self.target[0])
        self.plt_target.set_ydata(self.target[1])

        # render image
        pix_dimensions = (int(self.fig.get_size_inches()[0]*dpi),
                          int(self.fig.get_size_inches()[1]*dpi), -1)
        io_buf = io.BytesIO()
        self.fig.savefig(io_buf, format='raw', dpi=dpi)
        io_buf.seek(0)
        img = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8), newshape=pix_dimensions)
        io_buf.close()

        if show_plot:
            plt.imshow(img)
            plt.axis('off')

        return img


# class Arm(Env):
#
#     def __init__(self):
#         self.env = suite.load('reacher', 'easy')
#         self.env.reset()  # initial randomization appears to require this line... :/
#         self.max_steps = self.env._step_limit
#         self.Q = np.diag((0,0,1,1,0,0))  # only weight distance from target
#         self.R = np.zeros((6,6))         # no control costs for now
#
#     def reset(self):
#         pass
#
#     def step(self, action):
#         state = self.parse_timestep(self.env.step(action))
#         return state
#
#     def cost(self, state, action):
#         # J(t) = .5*(x^TQx + uTRt)
#         # J =  .5 * np.dot(self.state,  np.matmul(self.Q, self.s self.state))
#         # J += .5 * np.dot(self.action, np.matmul(self.R, self.action))
#         return 0
#
#     def cost_final(self, state):
#         J =  .5 * np.dot(self.state,  np.matmul(self.Q, self.state))
#
#     def set_state(self, state):
#         """ set state without changing target position """
#         with self.env.physics.reset_context():
#             self.env.physics.data.qpos[:] = state[:2]
#             self.env.physics.data.qvel[:] = state[-2:]
#
#     @staticmethod
#     def parse_timestep(time_step):
#         """ extract state (p0, p1, to_target0, to_target1, v0, v1) """
#         return np.concatenate(list(time_step.observation.values()))
#
#     def render(self):
#         return self.env.physics.render(camera_id=0)
