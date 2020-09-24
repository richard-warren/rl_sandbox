from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
from dm_control import suite
import numpy as np
import matplotlib
import ipdb
import io



class Env:

    def __init__(self):
        pass

    def simulate(self, state, action):
        """ get subsequent state from (state, action) """
        original_state = self.state.copy()
        self.state = state.copy()
        state_next = self.step(action)
        self.state = original_state.copy()  # put state back where it was
        return state_next

    def rollout(self, actions):
        """ compute states, costs, and cost derivatives for trajectory controlled by `actions` """
        # todo: check that doesn't exceed max_steps
        states, costs, costs_derivs = [], [], []
        states.append(self.reset(reset_target=False))

        for action in actions:
            cost, cost_derivs = self.cost(states[-1], action)
            costs.append(cost)
            costs_derivs.append(cost_derivs)
            states.append(self.step(action))

        cost, cost_derivs = self.cost_final(states[-1])
        costs.append(cost)
        costs_derivs.append(cost_derivs)

        return states, costs, costs_derivs

    def state_derivs(self, state, action):
        """ compute derivates of state wrt states and controls (f_x, f_u) """
        state = np.array(state, dtype='float64')
        action = np.array(action, dtype='float64')

        sim_state = lambda x: self.simulate(x, action)  # keeping action constant
        f_x = self.finite_differences(sim_state, state)

        sim_action = lambda x: self.simulate(state, x)  # keeping state constant
        f_u = self.finite_differences(sim_action, action)

        return dict(f_x=f_x, f_u=f_u)

    @staticmethod
    def finite_differences(fcn, x, eps=1e-4):  # if eps too small derivs may be zero
        """ estimate gradient of fcn wrt x via finite differences """
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

    def show(self):
        plt.imshow(self.render())
        plt.axis('off')


class PointMass(Env):
    """ point mass in plane with target. cost is distant to target """

    def __init__(self, dt=.05, arena_size=(1,1), mass=1, max_time=10):
        self.mass = mass
        self.dt = dt
        self.xlim = (-arena_size[0]/2, arena_size[0]/2)
        self.ylim = (-arena_size[1]/2, arena_size[1]/2)
        self.max_steps = int(max_time // dt)
        self.reset()  # set initial state and target positions

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
        """ reset point to initial state and target to random position """
        self.state = np.array([0,0,0,0], dtype='float64')  # (pos_x, pos_y, vel_x, vel_y)
        if reset_target:
            self.target = np.random.uniform((self.xlim[0], self.ylim[0]),
                                            (self.xlim[1], self.ylim[1]))
        return self.state.copy()

    def step(self, action):
        """ advance state via F=ma """
        self.state[2:] += np.array(action, dtype='float64') * self.dt
        self.state[:2] += self.state[2:] * self.dt
        return self.state.copy()

    def cost(self, state, action):
        """ compute cost and cost derivaties """
        cost = 0.5 * ((state[:2] - self.target)**2).sum()
        derivs = {
            'l_x':  np.concatenate(((state[:2]-self.target), [0,0])),
            'l_u':  np.zeros(2),
            'l_ux': np.zeros((2,4)),
            'l_xx': np.diag((1,1,0,0)),
            'l_uu': np.zeros((2,2))
        }
        return cost, derivs

    def cost_final(self, state):
        """ compute final cost and final cost derivaties """
        cost = 0.5 * ((state[:2] - self.target)**2).sum()
        derivs = {
            'l_x':  np.concatenate(((state[:2]-self.target), [0,0])),
            'l_u':  np.zeros(2),
            'l_ux': np.zeros((2,4)),
            'l_xx': np.diag((1,1,0,0)),
            'l_uu': np.zeros((2,2))
        }
        return cost, derivs

    def render(self, dpi=200):
        """ render image of current state """
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

        return img


class Arm(Env):
    """ two link arm. wrapper from dm_control `reacher` """

    def __init__(self, control_wgt=0):
        self.env = suite.load('reacher', 'easy')
        self.reset()
        self.max_steps = 250 # int(self.env._step_limit) !!! temp
        self.dt = self.env.physics.timestep()
        self.control_wgt = control_wgt

    @property
    def state(self):
        """ get state (pos0, pos1, vel0, vel1) """
        return self.env.physics.get_state()

    @state.setter
    def state(self, state):
        """ set state without changing target position """
        with self.env.physics.reset_context():
            self.env.physics.data.qpos[:] = state[:2].copy()
            self.env.physics.data.qvel[:] = state[2:].copy()

    def reset(self, reset_target=True):
        """ reset state (target position randomized but arm set to default state) """

        if not reset_target:
            target_pos = (self.env.physics.named.model.geom_pos['target', 'x'],
                          self.env.physics.named.model.geom_pos['target', 'y'])
            self.env.reset()

            # reset target
            with self.env.physics.reset_context():
                self.env.physics.named.model.geom_pos['target', 'x'] = target_pos[0]
                self.env.physics.named.model.geom_pos['target', 'y'] = target_pos[1]
        else:
            self.env.reset()

        # reset arm state
        self.state = [0,0,0,0]

        return self.state.copy()

    def step(self, action):
        """ advance state via physics engine """
        self.env.step(action)
        return self.state

    def cost(self, state, action, compute_derivs=True):
        """ compute cost and cost derivaties """
        self.state = state
        cost = 0.5 * self.env.physics.finger_to_target_dist()**2

        if compute_derivs:
            fcn_x = lambda x: self.cost(x, action, compute_derivs=False)[0]
            l_x = self.finite_differences(fcn_x, state)

            fcn_xx = lambda x: self.finite_differences(fcn_x, x)
            l_xx = self.finite_differences(fcn_xx, state)

            derivs = {
                'l_x':  l_x,
                'l_u':  2*np.array(action)*self.control_wgt,
                'l_ux': np.zeros((2,4)),
                'l_xx': l_xx,
                'l_uu': np.full((2,2), 2*self.control_wgt),
            }
        else:
            derivs = None

        return cost, derivs

    def cost_final(self, state, compute_derivs=True):
        """ compute final cost and final cost derivaties """
        self.state = state
        cost = 0.5 * self.env.physics.finger_to_target_dist()**2

        if compute_derivs:
            fcn_x = lambda x: self.cost_final(x, compute_derivs=False)[0]
            l_x = self.finite_differences(fcn_x, state)

            fcn_xx = lambda x: self.finite_differences(fcn_x, x)
            l_xx = self.finite_differences(fcn_xx, state)

            derivs = {
                'l_x':  l_x,
                'l_u':  np.zeros(2),
                'l_ux': np.zeros((2,4)),
                'l_xx': l_xx,
                'l_uu': np.zeros((2,2))
            }
        else:
            derivs = None

        return cost, derivs

    def render(self):
        """ render image of current state """
        return self.env.physics.render(camera_id=0)


class Test:
    def __init__(self):
        self._x = 0