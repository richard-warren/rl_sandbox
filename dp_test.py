import gym
import rickgrid
from rickgrid.mazes import mazes
import numpy as np
import matplotlib.pyplot as plt
from rickgrid import Agents
import ipdb
import time

# make environment
env = mazes['two_terminal']
env.render()

## dynamic programming

max_steps = 50
v = np.zeros((env.observation_space.n, max_steps+1))  # (state X time) max value of each state at each time
a = np.zeros((env.observation_space.n, max_steps), dtype=int)  # best actions to take in each state at each time

for i in reversed(range(max_steps)):
    for s in range(env.observation_space.n):
        v_next = env.R[s] + v[env.P[s],i+1]
        a[s,i] = np.argmax(v_next)
        v[s,i] = np.max(v_next)

## follow the plan, stan
env.reset()
done = False

for i in range(max_steps):
    done = env.step(a[env.state,i])[2]
    env.render(); time.sleep(.1)
    if done:
        break



