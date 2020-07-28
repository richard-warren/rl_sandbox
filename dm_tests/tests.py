# for reloading modules
%load_ext autoreload
%autoreload 2
##
# todo: basic training loop // evaluate policy? // clean up expand_dims parts // q visualization

# imports
import numpy as np
from dm_control import suite
import matplotlib.pyplot as plt
from tqdm import tqdm
import dm_tests.Agents

# functions
def get_avg_return(agent, env, iterations=1, max_time=10):
    returns = []
    for i in range(iterations):
        time_step = env.reset()
        episode_return = 0
        while not time_step.last() and env.physics.time()<max_time:
            action = agent.select_action(time_step, epsilon=0)
            time_step = env.step(action)
            episode_return += time_step.reward
        returns.append(episode_return)
    return sum(returns) / len(returns)


# fill buffer with random actions
def initialize_buffer(agent, env):
    print('initializing replay buffer...')
    time_step = env.reset()
    for i in tqdm(range(agent.buffer_length)):
        action = agent.select_action(time_step, epsilon=1)
        time_step_next = env.step(action)
        agent.add_experience(time_step, action, time_step_next)
        time_step = time_step_next


# initialize q to output optimistic values
def train_optimistic_q(agent, target_q = 100, iterations=1000, batch_size=128):
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




## train

# settings
action_dim = 3
iterations = 10000
steps_per_iteration = 4  # actions to take before updating q
batch_size = 64
q_update_interval = 10  # in batches
eval_interval = 2000
epsilon_start = 1
epsilon_final = .1
target_q = None

# choose task
env = suite.load('cartpole', 'balance')  # success
# env = suite.load('cartpole', 'balance_sparse')  # success
# env = suite.load('cartpole', 'swingup')  # success
# env = suite.load('cartpole', 'swingup_sparse')  # success!
time_step = env.reset()

# make agent
agent = dm_tests.Agents.Agent(env.observation_spec(), env.action_spec(),
                              action_dim=action_dim,
                              q_update_interval=q_update_interval)
# initialize_buffer(agent, env)
if target_q is not None:
    train_optimistic_q(agent, target_q=target_q)

##
print('training agent...')

for i in tqdm(range(iterations)):
    for j in range(steps_per_iteration):
        # epsilon_temp = epsilon_start - (i / iterations) * (epsilon_start - epsilon_final)
        epsilon_temp = .5
        action = agent.select_action(time_step, epsilon=epsilon_temp)
        time_step_next = env.step(action)
        agent.add_experience(time_step, action, time_step_next)
        time_step = time_step_next

    agent.update(batch_size=batch_size)

    if (i+1) % eval_interval == 0:
        avg_return = get_avg_return(agent, env)
        print('iteration {}, avg return {}, epsilon {}'.format(i+1, avg_return, epsilon_temp))



## show rollout

time_step = env.reset()
imshow = plt.imshow(env.physics.render(camera_id=0))

while not time_step.last():
    action = agent.select_action(time_step, epsilon=0)
    time_step = env.step(action)
    imshow.set_data(env.physics.render(camera_id=0))
    plt.pause(.001)
print('all done!')

## test random initialization



