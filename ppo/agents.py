from jax.scipy.stats import multivariate_normal
from jax.lax import stop_gradient
from functools import partial
from ppo.utils import mlp
import jax.numpy as jnp
import numpy as np
import jax


method_jit = partial(jax.jit, static_argnums=(0,))  # jit that works with class methods


class ActorCritic():
    ''' todo: make super class, subclasses for discrete, cont, shared, not shared params, etc '''

    def __init__(self, state_dim, action_dim, std=1, seed=0,
                 actor_hlayers=(32,16), critic_hlayers=(32,16)):
        ''' todo: should std be object property? '''

        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.std = std

        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, num=2)
        self.actor, actor_params   = mlp(keys[0], state_dim, action_dim, actor_hlayers)
        self.critic, critic_params = mlp(keys[1], state_dim, 1, critic_hlayers)
        self.params = {'actor': actor_params, 'critic': critic_params}

    @method_jit
    def act(self, key, actor_params, state: np.array):
        action_mean = self.actor(actor_params, state)
        cov = jnp.diag(jnp.repeat(self.std, self.action_dim))

        action = jax.random.multivariate_normal(key, action_mean, cov)
        action_prob = multivariate_normal.pdf(action, action_mean, cov)

        return stop_gradient(action), stop_gradient(action_prob)

    @method_jit
    def evaluate(self, critic_params, states: np.array):
        return jnp.squeeze(self.critic(critic_params, states))

    @method_jit
    def get_action_probs(self, actor_params, states: np.array, actions: np.array):
        # todo: make std an arg so modifiable during after compilation?
        action_means = self.actor(actor_params, states)
        cov = jnp.diag(jnp.repeat(self.std, self.action_dim))
        action_probs = multivariate_normal.pdf(actions, action_means, cov)

        return action_probs
