from dm_control_tests import agents
from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
from dm_control import suite
import numpy as np
from dm_control_tests import train_utils
import matplotlib
import copy



# show rollout
def show_rollout(agent, env, epsilon=0):
    env = copy.deepcopy(env)  # don't mess with state of the original environment
    plt.figure()
    time_step = env.reset()
    imshow = plt.imshow(env.physics.render(camera_id=0))
    episode_return = 0
    while not time_step.last():
        action = agent.select_action(time_step, epsilon=epsilon)
        time_step = env.step(action)
        episode_return += time_step.reward
        imshow.set_data(env.physics.render(camera_id=0))
        plt.pause(.001)
    print('episode return: {:.2f}'.format(episode_return))


# show rollout
def show_rollout_jupyter(agent, env, epsilon=0, framerate=30, max_time=None, rand_seed=None):
    env = copy.deepcopy(env)  # don't mess with state of the original environment
    if rand_seed is not None:
        train_utils.rand_seed_reset(env, rand_seed)
    if max_time is None:
        max_time = env._step_limit * env.physics.timestep() * env._n_sub_steps
    time_step = env.reset()

    # collect frames
    frames = []
    while not time_step.last() and env.physics.time()<max_time:
        time_step = env.step(agent.select_action(time_step, epsilon=epsilon))
        frames.append(env.physics.render(camera_id=0))

    return display_video(frames, framerate=framerate)


# show videos inline given frames
# modified from: https://colab.research.google.com/github/deepmind/dm_control/blob/master/tutorial.ipynb#scrollTo=gKc1FNhKiVJX
def display_video(frames, framerate=30, is_plot=False, imshow_args={}, xlabel=None, ylabel=None):
    dpi=70
    height, width = frames[0].shape[:2]

    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(5,5) if is_plot else (width/dpi, height/dpi), dpi=dpi);

    matplotlib.use(orig_backend)  # Switch back to the original backend.
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)

    if not is_plot:
        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.set_position([0, 0, 1, 1])
    else:
        ax.set_position([.15, .15, .8, .8])

    im = ax.imshow(frames[0], **imshow_args, aspect='auto');
    def update(frame):
      im.set_data(frame)
      return [im]

    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)

    return HTML(anim.to_html5_video())


