# for reloading modules
%load_ext autoreload
%autoreload 2

"""
todo:
- get working again
  - is agent being modified within functions?
  - check deep copy of timestep
  - check function scope naming
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




## train

# settings
action_dim = 3
updates = 10000
steps_per_update = 4  # actions to take before updating q (there are 1000 steps in an episode)
batch_size = 64
q_update_interval = 100  # in updates
eval_interval = 1000  # steps before next update
epsilon_start = 1
epsilon_final = .1
buffer_length = 10000
target_q = None
buffer_init = True  # whether to initialize replay_buffer with random actions
learning_rate = .01

# choose task
# env = suite.load('cartpole', 'balance')  # success
# env = suite.load('cartpole', 'balance_sparse')  # success (1000 max)
env = suite.load('cartpole', 'swingup')  # success
# env = suite.load('cartpole', 'swingup_sparse')  # success!

# make agent
agent = Agents.Agent(env.observation_spec(), env.action_spec(), action_dim=action_dim,
                     q_update_interval=q_update_interval, buffer_length=buffer_length, learning_rate=learning_rate)

if buffer_init:
    utils.initialize_buffer(agent, env)

if target_q is not None:
    utils.train_optimistic_q(agent, target_q=target_q)

##
print('training agent...')
time_step = env.reset()
for i in tqdm(range(updates)):
    for j in range(steps_per_update):
        epsilon_temp = epsilon_start - (i / updates) * (epsilon_start - epsilon_final)
        action = agent.select_action(time_step, epsilon=epsilon_temp)
        time_step_next = env.step(action)
        agent.add_experience(time_step, action, time_step_next)
        time_step = time_step_next

    agent.update(batch_size=batch_size)

    if (i+1) % eval_interval == 0:
        avg_return = utils.get_avg_return(agent, env)
        print('iteration {}, avg return {}, epsilon {}'.format(i+1, avg_return, epsilon_temp))
        time_step = env.reset()



## show rollout
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