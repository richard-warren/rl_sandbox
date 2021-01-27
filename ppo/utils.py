from jax.experimental import stax
import numpy as np


def mlp(key, input_dim, output_dim, hidden_layers=(64,32)):
    ''' make multilayer perceptron '''
    layers = []
    for hl in hidden_layers:
        layers += [stax.Dense(hl), stax.Tanh]
    layers.append(stax.Dense(output_dim))

    init_fun, apply_fun = stax.serial(*layers)
    params = init_fun(key, (-1, input_dim))[1]

    return apply_fun, params


def squash_actions(action, mins: np.array, maxes: np.array):
    ''' each row of action is an action vector '''
    rngs = maxes - mins
    squashed = (rngs / (1 + np.exp(-np.squeeze(action)))) + mins
    return squashed
