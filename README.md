# gridworld
This is an implementation of gridworld along with some control methods. See [the demo notebook](https://github.com/rwarren2163/rl_sandbox/blob/master/demo.ipynb) for usage and experiment results. 

#### todo
- [ ] notebook ((q, mc, dp, exact dp), (experience replay effects), (qinit, rand stars, q with noise, ucb))
- [ ] episode movie?
- [ ] refactor RickGrid, which is getting pretty messy
- [X] make max_steps an attribute of environment instead of agent! this makes more sense...
- [X] nicer way of saving default mazes
- [X] read, implement exact dynamic programming
- [X] proper transition matrices
- [X] nicer online update during rollout approach
- [X] class refactoring
- [X] default maze library
- [X] random start state option
- [X] value function visualization
- [X] policy visualization
- [X] classes for solutions
- [X] jupyter notebook demo

#### demo
- [X] gridworld
- [X] model-free control
    - [X] q learning
    - [X] mc
- [X] model-based control
    - [X] approximate dp
    - [X] exact dp
- explore vs. exploit!
    - [X] optimistic initialization
    - [X] random starts
    - [X] exact DP
    - adding noise
- control method comparison? 