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
- other
    - wall time?
    - exhaustive sweep over tasks?
    - make sure gamma appears in evaluations as well...

# demo outline
- dqn on dm_control
- setup
  - description + outline
  - inits
  - hyperparams
  -training details
- solving tasks
  - pendulum (value fcn plots, training, vid)
  - swingup (training, vid)
  - ball_in_cup (training, vid)
- double dqn
  - action value density plots for pendulum
  - density plots using double dqn
- increasing exploration with optimistic initializations
  - fraction state space explored, or entropy of distro, over time?
  - evolution of action value function...