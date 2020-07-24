# for reloading modules
%load_ext autoreload
%autoreload 2

# todo: basic training loop // evaluate policy? // clean up expand_dims parts

## imports
import numpy as np
from dm_control import suite
import matplotlib.pyplot as plt
import tqdm

# rick imports
import dm_tests.Agents

env = suite.load('cartpole', 'balance')


##
def get_avg_return(agent, env, iterations=1):
    # todo: should i be incorporating discounting here?

    returns = []
    for i in range(iterations):
        print(i)
        time_step = env.reset()
        episode_return = 0
        while not time_step.last():
            print(env.physics.time())
            action = agent.select_action(time_step)
            time_step = env.step(action)
            episode_return += time_step.reward
        returns.append(episode_return)

    return sum(returns) / len(returns)


## train

# settings
action_dim = 32
iterations = 10000
steps_per_iteration = 4
batch_size = 32
q_update_interval = 100*batch_size

agent = dm_tests.Agents.Agent(env.observation_spec(), env.action_spec(),
                              action_dim=action_dim,
                              q_update_interval=q_update_interval)

time_step = env.reset()
for i in range(iterations):
    for j in range(steps_per_iteration):
        action = agent.select_action(time_step, epsilon=.5)
        time_step_next = env.step(action)
        agent.add_experience(time_step, action, time_step_next)
        time_step = time_step_next
    agent.update(batch_size=batch_size)


## show rollout

time_step = env.reset()
imshow = plt.imshow(env.physics.render(camera_id=0))

while env.physics.time()<4:
    action = agent.select_action(time_step, epsilon=0)
    time_step = env.step(action)
    imshow.set_data(env.physics.render(camera_id=0))
    plt.pause(.001)













