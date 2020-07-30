# for reloading modules
%load_ext autoreload
%autoreload 2

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




## train

# settings
action_dim = 2
episodes = 100  # 1000 steps per episode
steps_per_update = 4  # actions to take before updating q (there are 1000 steps in an episode)
batch_size = 64
q_update_interval = 100  # (updates) frequency of q_target updates
eval_interval = 10  # (episodes) check average return every eval_interval updates
epsilon_start = 1
epsilon_final = .1
epsilon_final_update = 50000  # (updates) epsilon_final is reach after this many updates
buffer_length = 10000  # (experiences)
target_q = None
learning_rate = .001

# choose task
# env = suite.load('cartpole', 'balance')  # success
# env = suite.load('cartpole', 'balance_sparse')  # success (1000 max)
env = suite.load('cartpole', 'swingup')  # success
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

# utils.show_rollout(agent, env)

##

utils.show_rollout(agent, env)


