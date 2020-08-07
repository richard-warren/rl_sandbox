# rl_sandbox

This repro contains reinforcement learning projects:
- `gridworld` contains a custom gridworld implementation build within the OpenAI Gym framework.
- `dm_control` contains tests of DeepMind Control suite problems.

# dm_control demo ideas
- q function accuracy with and without double dqn
- fraction state space explored (entropy of distribution) over training with and without optimistic q

# dqn todo
- make sure works on
  - pendulum
  - cartpole (balance, balance_sparse, swingup, swingup_sparse)
  - point mass
  - ball in cup
  
- wall time?
- exhaustive sweep over tasks?
- make sure gamma appears in evaluations as well...