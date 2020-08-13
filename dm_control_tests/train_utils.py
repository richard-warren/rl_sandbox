from concurrent.futures import ProcessPoolExecutor
from dm_control_tests import agents
from itertools import repeat
from dm_control import suite
from tqdm.auto import tqdm
import tensorflow as tf
import numpy as np
import pickle
import random
import shutil
import copy
import os


# reset random seeds
def rand_seed_reset(env, i):
    random.seed(i)
    np.random.seed(i)
    tf.random.set_seed(i)
    env.task.random.seed(i)


# disable GPUs for tensorflow (CPU is faster for small networks/batches on my machine)
def disable_gpu():
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'


# compute average return by sampling episodes
def get_avg_return(agent, env, episodes=5, epsilon=.05):
    env = copy.deepcopy(env)  # don't mess with state of original env
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


# fill replay buffer with random actions
def initialize_buffer(agent, env, verbose=False):
    env = copy.deepcopy(env)  # don't mess with state of original env
    if verbose: print('initializing replay buffer...')
    time_step = env.reset()
    for _ in tqdm(range(agent.buffer_length)) if verbose else range(agent.buffer_length):
        action = agent.select_action(time_step, epsilon=1)
        time_step_next = env.step(action)
        agent.add_experience(time_step, action, time_step_next)
        time_step = time_step_next
        if time_step_next.last():
            env.reset()


# initialize q to output values close to target_q throughout state space
def train_optimistic_q(agent, target_q=100, iterations=1000, batch_size=128, verbose=True):
    if verbose: print('initializing q with optimistic values...')

    # get statistics of replay buffer
    observations = np.vstack([np.vstack((i[0],i[3])) for i in agent.replay_buffer])
    obs_min = np.min(observations, axis=0)
    obs_ptp = np.ptp(observations, axis=0)

    # sample from hyper-rectangle spanning buffer data
    def get_random_samples(num_samples=32):
        smp = np.random.uniform(size=(num_samples,agent.observation_dim))
        return np.multiply(smp, obs_ptp) + obs_min

    # get average output of network prior to training (on uniformly sampled random inputs)
    if verbose:
        def get_avg_value(num_samples=100):
            smp = get_random_samples(num_samples)
            values = agent.q.predict(smp)
            return np.mean(values)

        print('pre training avg value: {:.2f}'.format(get_avg_value()))

    for _ in tqdm(range(iterations)) if verbose else range(iterations):
        agent.q.fit(get_random_samples(batch_size), np.ones(batch_size)*target_q, verbose=False)
    agent.q_target.set_weights(agent.q.get_weights())

    if verbose:
        print('post training avg value: {:.2f}'.format(get_avg_value()))

    # reset optimizer state
    # todo: should save original compile args to make sure none are missing here...
    agent.q.compile(loss=agent.q.loss, optimizer=agent.q.optimizer)


# train DQN agent
def train(agent, env, episodes=100, action_repeats=4, steps_per_update=4, gamma=.99, batch_size=64,
          epsilon_start=1, epsilon_final=.1, epsilon_final_episode=50,
          eval_interval=10, eval_epsilon=.05, eval_episodes=5, verbose=True, progress_bar=True, callback=None):

    if progress_bar: print('training agent...')
    avg_return, returns = get_avg_return(agent, env, epsilon=eval_epsilon, episodes=eval_episodes)
    episode_num, all_returns = [0], [returns]
    callback_returns = []

    for i in tqdm(range(episodes)) if progress_bar else range(episodes):
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
            episode_num.append(i+1)
            all_returns.append(returns)
            if verbose: print('iteration {:4d}, avg return {:4.1f}'.format(i+1, avg_return))
            if callback is not None:
                callback_returns.append(callback(agent, env))

    if callback is None:
        return episode_num, all_returns
    else:
        return episode_num, all_returns, callback_returns


# train a single agent on a particular domain and task
def create_and_train_agent(domain_and_task, agent_args, train_args, optimistic_q=None,
                           save_path=None, verbose=True, progress_bar=True):
    env = suite.load(*domain_and_task)
    agent = agents.Agent(env.observation_spec(), env.action_spec(), **agent_args)
    initialize_buffer(agent, env, verbose=False)
    if optimistic_q is not None:
        train_optimistic_q(agent, target_q=optimistic_q, iterations=1000, batch_size=128, verbose=verbose)
    episode_num, returns = train(agent, env, **train_args, verbose=verbose, progress_bar=progress_bar)

    # save
    if save_path is not None:
        os.mkdir(save_path)
        agent.q.save(os.path.join(save_path, 'q_network'))
        agent.q, agent.q_target = None, None  # so can be pickled
        metadata = {
            'domain_and_task': domain_and_task,
            'agent_args': agent_args,
            'train_args': train_args,
            'optimistic_q': optimistic_q
        }
        with open(os.path.join(save_path, 'agent'), 'wb') as file:
            pickle.dump(agent, file)
        with open(os.path.join(save_path, 'metadata'), 'wb') as file:
            pickle.dump(metadata, file)
    return episode_num, returns


# train multiple agents in parallel on a particular domain and task
def train_agents_parallel(domain_and_task, agent_args, train_args,
                          optimistic_q=None, n_agents=12, n_workers=12,
                          save_dir=None, overwrite=False):
    if save_dir is not None:
        if overwrite and os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        paths = ['{}/agent{:03d}'.format(save_dir, i) for i in range(n_agents)]
    else:
        paths = repeat(None, n_agents)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(create_and_train_agent,
                               repeat(domain_and_task, n_agents),
                               repeat(agent_args),
                               repeat(train_args),
                               repeat(optimistic_q),
                               paths,
                               repeat(False),
                               [True] + list(repeat(False, n_agents - 1)))  # only progress bar for first thread
        results = [r for r in results]
    episode_num = results[0][0]
    avg_returns = np.array([np.array(r[1]).mean(1) for r in results])

    if save_dir is not None:
        training_data = {'episode_num': episode_num, 'avg_returns': avg_returns}
        with open(os.path.join(save_dir, 'training_data'), 'wb') as file:
            pickle.dump(training_data, file)

    return episode_num, avg_returns


def load_agent(save_path):
    with open(os.path.join(save_path, 'agent'), 'rb') as file:
        agent = pickle.load(file)
    agent.q = tf.keras.models.load_model(os.path.join(save_path, 'q_network'))
    agent.q_target = copy.copy(agent.q)  # todo: should really save and reload both q and q_target
    with open(os.path.join(save_path, 'metadata'), 'rb') as file:
        metadata = pickle.load(file)
    return agent, metadata
