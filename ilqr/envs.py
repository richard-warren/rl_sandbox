from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
from dm_control import suite
import numpy as np
import matplotlib
import ipdb
import io



class Env:

    def simulate(self, state, action):
        """ get subsequent state from (state, action) """
        original_state = self.state
        self.state = state
        state_next = self.step(action)
        self.state = original_state  # put state back where it was
        return state_next

    def rollout(self, actions):
        """ compute states, costs, and cost derivatives for trajectory controlled by `actions` """
        states, costs, costs_derivs = [], [], []
        states.append(self.reset(reset_target=False))

        for action in actions:
            cost = self.cost(states[-1], action)
            cost_derivs = self.cost_derivs(states[-1], action)
            costs.append(cost)
            costs_derivs.append(cost_derivs)
            states.append(self.step(action))

        cost = self.cost_final(states[-1])
        cost_derivs = self.cost_final_derivs(states[-1])
        costs.append(cost)
        costs_derivs.append(cost_derivs)

        return states, costs, costs_derivs

    def state_derivs(self, state, action):
        """ compute derivates of state wrt states and controls (f_x, f_u) """
        xu = np.concatenate((np.array(state), np.array(action)))
        state_derivs = lambda xu: self.simulate(xu[:4], xu[4:])
        J = self.finite_differences(state_derivs, xu)
        return dict(f_x=J[:,:4], f_u=J[:,4:])

    def cost_derivs(self, state, action):
        """ compute cost and cost derivatives """
        state = np.array(state, dtype='float64')
        action = np.array(action, dtype='float64')
        original_state = self.state
        self.state = state
        ix = len(state)

        # first order
        xu = np.concatenate((np.array(state), np.array(action)))
        xu_cost = lambda xu: self.cost(xu[:ix], xu[ix:])
        J = self.finite_differences(xu_cost, xu)
        l_x = J[:ix]
        l_u = J[ix:]

        # second order
        xu_derivs = lambda xu: self.finite_differences(xu_cost, xu)
        J = self.finite_differences(xu_derivs, xu)
        l_xx = J[:ix,:ix]
        l_uu = J[ix:,ix:]
        l_ux = J[ix:,:ix]

        self.state = original_state  # put state back where it was
        return dict(l_x=l_x, l_u=l_u, l_ux=l_ux, l_xx=l_xx, l_uu=l_uu)

    def cost_final_derivs(self, state):
        """ compute final cost and final cost derivaties """
        state = np.array(state)
        original_state = self.state
        self.state = np.array(state)

        x_cost = lambda x: self.cost_final(x)
        l_x = self.finite_differences(x_cost, state)
        x_derivs = lambda v: self.finite_differences(self.cost_final, v)
        l_xx = self.finite_differences(x_derivs, state)

        self.state = original_state  # put state back where it was
        return dict(l_x=l_x, l_xx=l_xx)

    @staticmethod
    def finite_differences(fcn, x, eps=1e-4):  # if eps too small derivs may be zero
        """ estimate gradient of fcn wrt x v fia finite differences """
        # todo: vectorize (assuming fcn is vectorized)
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

    def __init__(self, dt=.05, arena_size=(1,1), mass=1, max_time=10, initial_state=[0,0,0,0],
                 control_wgt=1e-4, state_wgt=1):
        self.mass = mass
        self.dt = dt
        self.xlim = (-arena_size[0]/2, arena_size[0]/2)
        self.ylim = (-arena_size[1]/2, arena_size[1]/2)
        self.max_steps = int(max_time // dt)
        self.initial_state = initial_state
        self.control_wgt = control_wgt
        self.state_wgt = state_wgt
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

    def cost(self, state, action):
        cost  = np.linalg.norm(state[:2] - self.target)**2 * self.state_wgt
        cost += sum(np.array(action)**2)                   * self.control_wgt
        return cost

    def cost_final(self, state):
        cost  = np.linalg.norm(state[:2] - self.target)**2 * self.state_wgt
        return cost

    def reset(self, reset_target=True):
        """ reset point to initial state and target to random position """
        self.state = np.array(self.initial_state, dtype='float64')  # (pos_x, pos_y, vel_x, vel_y)
        if reset_target:
            self.target = np.random.uniform((self.xlim[0], self.ylim[0]),
                                            (self.xlim[1], self.ylim[1]))
        return self.state.copy()

    def step(self, action):
        """ advance state via F=ma """
        self.state[2:] += np.array(action) * self.dt
        self.state[:2] += self.state[2:] * self.dt
        return self.state.copy()

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

    def __init__(self, initial_state=[np.pi/2,0,0,0], max_steps=None,
                 control_wgt=1e-4, state_wgt=1):
        self.initial_state = initial_state
        self.env = suite.load('reacher', 'hard')
        # self.env = suite.load('point_mass', 'easy')
        self.control_wgt = control_wgt
        self.state_wgt = state_wgt
        self.max_steps = max_steps if max_steps is not None else int(self.env._step_limit)
        self.dt = self.env.physics.timestep()
        self.reset()

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

    @property
    def target(self):
        target = [self.env.physics.named.model.geom_pos['target', 'x'],
                  self.env.physics.named.model.geom_pos['target', 'y']]
        return target

    @target.setter
    def target(self, target):
        with self.env.physics.reset_context():
            self.env.physics.named.model.geom_pos['target', 'x'] = target[0]
            self.env.physics.named.model.geom_pos['target', 'y'] = target[1]

    def cost(self, state, action):
        original_state = self.state
        self.state = state
        cost  = self.env.physics.finger_to_target_dist()**2 * self.state_wgt
        cost += sum(np.array(action)**2)                    * self.control_wgt
        self.state = original_state  # return to original state
        return cost

    def cost_final(self, state):
        original_state = self.state
        self.state = state
        cost  = self.env.physics.finger_to_target_dist()**2 * self.state_wgt
        self.state = original_state  # return to original state
        return cost

    def reset(self, reset_target=True):
        """ reset state (target position randomized but arm set to default state) """
        original_target = self.target.copy()
        self.env.reset()
        if not reset_target:
            self.target = original_target
        self.state = self.initial_state

        return self.state.copy()

    def step(self, action):
        """ advance state via physics engine """
        sig = lambda x: (1 / (1 + np.exp(-4*x))) * 2 - 1  # squash between 0 and 1
        action = sig(np.array(action))
        self.env.step(action)
        self.env._step_count = 0  # clamp time to avoid env resets
        return self.state

    def render(self):
        """ render image of current state """
        return self.env.physics.render(camera_id=0)
