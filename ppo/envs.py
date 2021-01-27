import matplotlib.pyplot as plt
from dm_control import suite
import numpy as np
import io


class DmEnv():
    ''' dm_control environment wrapper '''

    def __init__(self, domain, task, add_time_to_state=True):
        self.env = suite.load(domain, task)
        self.max_steps = int(self.env._step_limit)
        self.add_time_to_state = add_time_to_state
        time_step = self.env.reset()
        self.action_dim = self.env.action_spec().shape[0]
        self.state_dim = len(self.process_time_step(time_step)[0])
        self.action_min = self.env.action_spec().minimum
        self.action_max = self.env.action_spec().maximum

    def step(self, action):
        timestep = self.env.step(action)
        return self.process_time_step(timestep)

    def reset(self):
        return self.process_time_step(self.env.reset())[0]  # only return state

    @property
    def step_count(self):
        return self.env._step_count

    def process_time_step(self, time_step):
        """ Convert 'dm_env._environment.TimeStep' to (state, reward, done) """
        state = np.hstack([v for v in time_step.observation.values()])
        if self.add_time_to_state:
            state = np.append(state, self.step_count / self.max_steps)
        return state, time_step.reward, time_step.last()

    def render(self):
        return self.env.physics.render(camera_id=0)


class PointMass():
    """ point mass in plane with target. cost is distant to target """
    # todo: random seed option

    def __init__(self, dt=.05, max_time=10, arena_size=(1,1), mass=1, target_pos=None,
                 initial_state=(0.,0.,0.,0.), action_min=[-1.,-1.], action_max=[1.,1.],
                 add_time_to_state=False):
        # todo: add_time_to_state option
        self.mass = mass
        self.dt = dt
        self.arena_size = np.array(arena_size)
        self.xlim = (-arena_size[0]/2, arena_size[0]/2)
        self.ylim = (-arena_size[1]/2, arena_size[1]/2)
        self.max_steps = int(max_time // dt)
        self.initial_state = np.array(initial_state)
        self.action_min = np.array(action_min)
        self.action_max = np.array(action_max)
        self.action_dim = 2
        self.state_dim = 4 + add_time_to_state
        self.step_count = 0
        self.target_pos = target_pos  # if provided, always return target to this pos
        self.add_time_to_state = add_time_to_state
        self.reset()  # set initial state and target positions

        # graphic objects
        sz = arena_size
        self.fig = plt.figure(figsize=(2*(sz[0]/sz[1]), 2))
        self.ax = plt.axes(xlim=(-sz[0]*.6, sz[0]*.6), ylim=(-sz[1]*.6, sz[1]*.6))
        plt.axis('off')

        self.plt_target = plt.plot(
            self.target[0], self.target[1], marker='x', ms=10,
            markeredgewidth=2, markeredgecolor=(1,0,0))[0]
        self.plt_circle = plt.plot(
            self.state[0], self.state[1], marker='o', ms=10,
            alpha=.5, color='blue')[0]

        plt.plot([-sz[0]/2, sz[0]/2, sz[0]/2, -sz[0]/2, -sz[0]/2],
                 [sz[1]/2, sz[1]/2, -sz[1]/2, -sz[1]/2, sz[1]/2],
                 color='black', linewidth=4)  # arena walls
        plt.close()

    def reset(self):
        """ reset point to initial state and target to random position """
        self.step_count = 0

        self.state = np.array(self.initial_state, dtype='float64')  # (pos_x, pos_y, vel_x, vel_y)
        if self.add_time_to_state:
            self.state = np.append(self.state.copy(), 0)

        if self.target_pos is None:
            self.target = np.random.uniform((self.xlim[0], self.ylim[0]),
                                            (self.xlim[1], self.ylim[1]))
        else:
            self.target = self.target_pos

        return self.state.copy()

    def get_reward(self):
        ''' negative l2 distance from target '''
        l1_distances = (self.target - self.state[:2]) / self.arena_size  # normalize each dimension by its length
        return - np.sqrt(np.sum(l1_distances**2))

    def step(self, action):
        """ advance state via F=ma """
        self.step_count += 1
        done = self.step_count > self.max_steps # todo: throw error when stepping after done?

        self.state[2:4] += action * self.dt
        self.state[:2] += self.state[2:4] * self.dt
        self.state[:2] = np.clip(self.state[:2],  # don't let move outside walls
                                 (self.xlim[0], self.ylim[0]),
                                 (self.xlim[1], self.ylim[1]))
        if self.add_time_to_state:
            self.state[-1] = self.step_count/self.max_steps
        state = self.state.copy()  # todo: how to handle action lims? throw error? clip?

        reward = self.get_reward()

        return self.state.copy(), reward, done

    def render(self, dpi=200):
        """ render image of current state """

        # update graphics
        self.plt_circle.set_xdata(self.state[0])
        self.plt_circle.set_ydata(self.state[1])
        self.plt_target.set_xdata(self.target[0])
        self.plt_target.set_ydata(self.target[1])

        # render image
        pix_dimensions = (int(self.fig.get_size_inches()[1]*dpi),
                          int(self.fig.get_size_inches()[0]*dpi), -1)
        io_buf = io.BytesIO()
        self.fig.savefig(io_buf, format='raw', dpi=dpi)
        io_buf.seek(0)
        img = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8), newshape=pix_dimensions)
        io_buf.close()

        return img
