from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import copy
import ipdb


# get average episode return by sampling `iterations` episodes
def get_avg_return(agent, env, iterations=1, max_time=10, epsilon=0):
    env = copy.deepcopy(env)  # avoid modifying original object
    returns = []
    for i in range(iterations):
        time_step = env.reset()
        episode_return = 0
        while not time_step.last() and env.physics.time()<max_time:
            action = agent.select_action(time_step, epsilon=epsilon)
            time_step = env.step(action)
            episode_return += time_step.reward
        returns.append(episode_return)
    return sum(returns) / len(returns)


# fill buffer with random actions
def initialize_buffer(agent, env):
    env = copy.deepcopy(env)  # avoid modifying original object
    print('initializing replay buffer...')
    time_step = env.reset()
    for i in tqdm(range(agent.buffer_length)):
        action = agent.select_action(time_step, epsilon=1)
        time_step_next = env.step(action)
        agent.add_experience(time_step, action, time_step_next)
        time_step = time_step_next
        if time_step_next.last():
            env.reset()
        if time_step.reward is None:
            ipdb.set_trace()


# initialize q to output optimistic values
def train_optimistic_q(agent, target_q=100, iterations=1000, batch_size=128):
    print('initializing q with optimistic values...')

    # get statistics of replay buffer
    observations = np.vstack([np.vstack((i[0],i[3])) for i in agent.replay_buffer])
    obs_min = np.min(observations, axis=0)
    obs_ptp = np.ptp(observations, axis=0)

    # sample from hyper-rectangle spanning buffer data
    def get_random_samples(num_samples=32):
        smp = np.random.uniform(size=(num_samples,agent.observation_dim))
        return np.multiply(smp, obs_ptp) - obs_min

    def get_avg_value(num_samples=100):
        smp = get_random_samples(num_samples)
        values = agent.q.predict(smp)
        return np.mean(values)

    print('pre training avg value: {:.2f}'.format(get_avg_value()))
    for i in tqdm(range(iterations)):
        agent.q.fit(get_random_samples(batch_size), np.ones(batch_size)*target_q, verbose=False)
    print('post training avg value: {:.2f}'.format(get_avg_value()))


# show rollout
def show_rollout(agent, env):
    time_step = env.reset()
    imshow = plt.imshow(env.physics.render(camera_id=0))
    episode_return = 0
    while not time_step.last():
        action = agent.select_action(time_step, epsilon=0)
        time_step = env.step(action)
        episode_return += time_step.reward
        imshow.set_data(env.physics.render(camera_id=0))
        plt.pause(.001)
    print('episode return: {:.2f}'.format(episode_return))