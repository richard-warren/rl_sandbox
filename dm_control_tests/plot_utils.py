from dm_control_tests import agents
from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
from dm_control import suite
import numpy as np
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
def show_rollout_jupyter(agent, env, epsilon=0, framerate=30, max_time=None):
    env = copy.deepcopy(env)  # don't mess with state of the original environment
    time_step = env.reset()
    if max_time is None:
        max_time = env._step_limit * env.physics.timestep() * env._n_sub_steps

    # collect frames
    frames = []
    while not time_step.last() and env.physics.time()<max_time:
        time_step = env.step(agent.select_action(time_step, epsilon=epsilon))
        frames.append(env.physics.render(camera_id=0))

    return display_video(frames, framerate=framerate)


# show videos inline given frames
# borrowed from: https://colab.research.google.com/github/deepmind/dm_control/blob/master/tutorial.ipynb#scrollTo=gKc1FNhKiVJX
def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())


