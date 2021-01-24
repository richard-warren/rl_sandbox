# rl_sandbox
This repo contains implementations of several **reinforcement learning** algorithms. They are implemented within either:
- `gridworld`: a custom gridworld implementation built within the [OpenAI Gym](https://gym.openai.com/) framework.
  - Q-Learning
  - Monte Carlo Learning
  - Approximate Dynamic Programming
  - Exact Dynamic Programming
- `DQN`: physics-based simulations for continuous control tasks.
  - [DQN](https://www.nature.com/articles/nature14236)
  - [Double DQN](https://arxiv.org/abs/1509.06461)
- `iLQR`: iterative linear quadratic regular, [an optimal control theory algorithm](https://homes.cs.washington.edu/~todorov/papers/LiICINCO04.pdf).

# demos
- [gridworld demo](https://colab.research.google.com/github/richard-warren/rl_sandbox/blob/master/gridworld_demo.ipynb)
- [DQN demo](https://colab.research.google.com/github/richard-warren/rl_sandbox/blob/master/dqn_demo.ipynb)
- [iLQR demo](https://colab.research.google.com/github/richard-warren/rl_sandbox/blob/master/ilqr_demo.ipynb)
