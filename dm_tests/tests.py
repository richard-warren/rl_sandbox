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
import Agents, utils
import ipdb
import time




# train

# settings
action_dim = 2
updates = 10000  # 1000 steps per episode
steps_per_update = 2  # actions to take before updating q (there are 1000 steps in an episode)
batch_size = 32
q_update_interval = 100  # in updates
eval_interval = 1000  # check average return every eval_interval updates
epsilon_start = 1
epsilon_final = .1
epsilon_final_update = 2000  # epsilon_final is reach after this many updates
buffer_length = 10000
target_q = None
buffer_init = True  # whether to initialize replay_buffer with random actions
learning_rate = .01

# choose task
env = suite.load('cartpole', 'balance')  # success
# env = suite.load('cartpole', 'balance_sparse')  # success (1000 max)
# env = suite.load('cartpole', 'swingup')  # success
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
        epsilon_temp = epsilon_start - min(i/epsilon_final_update, 1) * (epsilon_start - epsilon_final)
        action = agent.select_action(time_step, epsilon=epsilon_temp)
        time_step_next = env.step(action)
        agent.add_experience(time_step, action, time_step_next)
        time_step = time_step_next

    agent.update(batch_size=batch_size)

    if (i+1) % eval_interval == 0:
        avg_return = utils.get_avg_return(agent, env)
        print('iteration {}, avg return {:.2f}, epsilon {:.2f}'.format(i+1, avg_return, epsilon_temp))
        time_step = env.reset()

utils.show_rollout(agent, env)


