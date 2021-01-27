from IPython.display import HTML
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib
import copy
import jax


def display_video(frames, framerate=30):
    '''
    show videos in jupyter notebook given list of frames
    modified from: https://colab.research.google.com/github/deepmind/dm_control/blob/master/tutorial.ipynb#scrollTo=gKc1FNhKiVJX
    '''

    dpi=70
    height, width = frames[0].shape[:2]
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(5,5), dpi=dpi);
    matplotlib.use(orig_backend)  # Switch back to the original backend.

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])

    im = ax.imshow(frames[0], aspect='auto');
    def update(frame):
      im.set_data(frame)
      return [im]

    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=1000/framerate, blit=True, repeat=False)
    return HTML(anim.to_html5_video());


def show_rollout(env, policy, framerate=100):
    # env = copy.deepcopy(env)  # don't mess with state of original env
    state = env.reset()
    done = False
    frames = []
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, env.max_steps)

    for k in keys:
        action = policy.act(k, policy.params['actor'], state)[0]
        state, _, done = env.step(action)
        frames.append(env.render())

    return display_video(frames, framerate=framerate)
