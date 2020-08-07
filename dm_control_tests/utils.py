from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib import animation
import matplotlib
from IPython.display import HTML



# get average episode return by sampling `iterations` episodes
def get_avg_return(agent, env, episodes=5, epsilon=.05):
    env = copy.deepcopy(env)  # don't mess with state of the original environment
    returns = []
    for i in range(episodes):
        time_step = env.reset()
        episode_return = 0
        while not time_step.last():
            action = agent.select_action(time_step, epsilon=epsilon)
            time_step = env.step(action)
            episode_return += time_step.reward
        returns.append(episode_return)
    return sum(returns) / len(returns), returns


# fill buffer with random actions
def initialize_buffer(agent, env):
    env = copy.deepcopy(env)  # don't mess with state of the original environment
    print('initializing replay buffer...')
    time_step = env.reset()
    for i in tqdm(range(agent.buffer_length)):
        action = agent.select_action(time_step, epsilon=1)
        time_step_next = env.step(action)
        agent.add_experience(time_step, action, time_step_next)
        time_step = time_step_next
        if time_step_next.last():
            env.reset()


# initialize q to output optimistic values throughout state space
def train_optimistic_q(agent, target_q=100, iterations=1000, batch_size=128):
    print('initializing q with optimistic values...')

    # get statistics of replay buffer
    observations = np.vstack([np.vstack((i[0],i[3])) for i in agent.replay_buffer])
    obs_min = np.min(observations, axis=0)
    obs_ptp = np.ptp(observations, axis=0)

    # sample from hyper-rectangle spanning buffer data
    def get_random_samples(num_samples=32):
        smp = np.random.uniform(size=(num_samples,agent.observation_dim))
        return np.multiply(smp, obs_ptp) + obs_min

    # get average output of network (on uniformly sampled random inputs)
    def get_avg_value(num_samples=100):
        smp = get_random_samples(num_samples)
        values = agent.q.predict(smp)
        return np.mean(values)

    print('pre training avg value: {:.2f}'.format(get_avg_value()))
    for i in tqdm(range(iterations)):
        agent.q.fit(get_random_samples(batch_size), np.ones(batch_size)*target_q, verbose=False)
    agent.q_target.set_weights(agent.q.get_weights())
    print('post training avg value: {:.2f}'.format(get_avg_value()))

    # reset optimizer state
    # todo: should save compile args as agent attribute to make sure none are missing here...
    agent.q.compile(loss=agent.q.loss, optimizer=agent.q.optimizer)


def train(agent, env, episodes=100, action_repeats=4, steps_per_update=4, gamma=.99, batch_size=64,
          epsilon_start=1, epsilon_final=.1, epsilon_final_episode=50,
          eval_interval=10, eval_epsilon=.1, eval_episodes=10, verbose=True):

    print('training agent...')
    episode_num, all_returns = [], []
    for i in tqdm(range(episodes)):
        time_step = env.reset()
        done = False
        action_counter = action_repeats
        step_counter = 0
        epsilon_temp = epsilon_start - min(i/epsilon_final_episode, 1) * (epsilon_start - epsilon_final)

        while not done:
            if action_counter==action_repeats:
                action = agent.select_action(time_step, epsilon=epsilon_temp)
                action_counter = 0
            action_counter += 1

            time_step_next = env.step(action)
            done = time_step_next.last()
            agent.add_experience(time_step, action, time_step_next)
            time_step = time_step_next

            step_counter += 1
            if step_counter == steps_per_update:
                agent.update(batch_size=batch_size, gamma=gamma)
                step_counter = 0

        if (i+1) % eval_interval == 0:
            avg_return, returns = get_avg_return(agent, env, epsilon=eval_epsilon, episodes=eval_episodes)
            episode_num.append(i)
            all_returns.append(returns)
            if verbose:
                print('iteration {}, avg return {:.2f}, epsilon {:.2f}, returns: {}'.format(
                    i + 1, avg_return, epsilon_temp, [int(x) for x in returns]))
    return episode_num, all_returns


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
        max_time = env._step_limit * env.physics.timestep()

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