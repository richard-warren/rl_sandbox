from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib


def display_video(frames, framerate=30):
    dpi=70
    height, width = frames[0].shape[:2]
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width/dpi, height/dpi), dpi=dpi);

    matplotlib.use(orig_backend)  # switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])

    im = ax.imshow(frames[0], aspect='auto');
    def update(frame):
      im.set_data(frame)
      return [im]

    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())

def show_rollout(env, actions):
    env.reset(reset_target=False)
    imgs = [env.render()]
    for action in actions:
        env.step(action)
        imgs.append(env.render())
    return display_video(imgs, framerate=(1/env.dt))

def plot_training(env, actions, history):
    ax = plt.subplots(1, 3, figsize=(14,4))[1];

    ax[0].plot(history['cost'], color='tab:blue', linewidth=3);
    ax[0].set_title('cost', fontweight='bold');
    ax[0].set_xlabel('iteration'); ax[0].set_ylabel('cost')
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    ax[1].plot(history['reg'], color='tab:blue', linewidth=3);
    ax[1].set_title('regularization', fontweight='bold');
    ax[1].set_xlabel('iteration'); ax[1].set_ylabel('regularization')
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)

    env.reset(reset_target=False)
    [env.step(action) for action in actions]
    ax[2].set_title('final state'); env.show();
