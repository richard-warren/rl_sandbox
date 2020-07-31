# for reloading modules
# %load_ext autoreload
# %autoreload 2

"""
todo:
- check gpu speed
"""

# imports
import numpy as np
import copy
from dm_control import suite
import matplotlib.pyplot as plt
from tqdm import tqdm
from dm_tests import Agents, utils
import ipdb
import time
import tensorflow as tf


print('DISABLING GPUs')
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'






## train

# settings
action_dim = 2
episodes = 100
steps_per_update = 4  # actions to take before updating q (increase this to cycle through episodes more quickly)
batch_size = 64
q_update_interval = 100  # (updates) frequency of q_target updates
eval_interval = 10  # (episodes) check average return every eval_interval updates
steps_per_episode = 1000  # don't change (this is a characteristic of the environment... (10 seconds / .01))
epsilon_start = 1
epsilon_final = .1
epsilon_final_update = (steps_per_episode /steps_per_update * episodes)*1  # (updates) epsilon_final is reach after this many updates
buffer_length = 10000
target_q = None  # use for "optimistic" initialize of q model
learning_rate = .001

# choose task
# env = suite.load('cartpole', 'balance')  # success
# env = suite.load('cartpole', 'balance_sparse')  # success (1000 max)
env = suite.load('cartpole', 'swingup')  # success (~600 is good)
# env = suite.load('cartpole', 'swingup_sparse')  # success!

# make agent
agent = Agents.Agent(env.observation_spec(), env.action_spec(), action_dim=action_dim,
                     q_update_interval=q_update_interval, buffer_length=buffer_length, learning_rate=learning_rate)
utils.initialize_buffer(agent, env)

if target_q is not None:
    utils.train_optimistic_q(agent, target_q=target_q)

##
print('training agent...')

for i in tqdm(range(episodes)):
    time_step = env.reset()
    done = False
    step_counter = 0

    while not done:
        epsilon_temp = epsilon_start - min(agent.total_updates / epsilon_final_update, 1) * (epsilon_start - epsilon_final)
        action = agent.select_action(time_step, epsilon=epsilon_temp)
        time_step_next = env.step(action)
        done = time_step_next.last()
        agent.add_experience(time_step, action, time_step_next)
        time_step = time_step_next

        step_counter += 1
        if step_counter==steps_per_update:
            agent.update(batch_size=batch_size)
            step_counter = 0

    if (i+1) % eval_interval == 0:
        avg_return = utils.get_avg_return(agent, env)
        print('iteration {}, avg return {:.2f}, epsilon {:.2f}'.format(i+1, avg_return, epsilon_temp))

##

utils.show_rollout(agent, env, epsilon=.0)



## plot q funcion (test)
import matplotlib.pyplot as plt
prediction_grid, axis_limits = agent.get_prediction_grid(bins=10)
axis_limits = np.array(axis_limits)
plt.close('all')
action_dim = 2
observation_dim = 5
fig = plt.figure(figsize=(16,6))
axes = fig.subplots(action_dim, observation_dim-1)

for i in range(observation_dim-1):
    for j in range(action_dim):
        mean_dims = tuple([d for d in range(observation_dim) if d<i or d>i+1])
        slice = prediction_grid.mean(axis=mean_dims)
        img = axes[j,i].imshow(slice[:,:,j], aspect='auto', extent=(axis_limits[0,i], axis_limits[1,i],
                                                                    axis_limits[0,i+1], axis_limits[1,i+1]))
        fig.colorbar(img, ax=axes[j,i])

