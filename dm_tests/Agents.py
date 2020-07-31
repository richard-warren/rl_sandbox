import random
import tensorflow as tf
import numpy as np
from collections import deque
import time
import ipdb

"""
todo:
speed things up
optimistic initialization (see if this process messes up subsequent rmsprop...)
fine tune optimizer and learning rate (should these be adjusted with optimistic inits?)
target across adjacent actions
make work on higher dimensional observation and action spaces

time (s):
- select action:        .017
- select action (np):   .001
- fit:                  .019
- predict batch:        .016
- prepare minibatch:    .0005
- step:                 .0002
"""


class Agent:

    def __init__(self, observation_spec, action_spec, action_dim=2, learning_rate=.001,
                 q_update_interval=100, buffer_length=10000):

        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.action_dim = action_dim
        self.actions = np.linspace(action_spec.minimum, action_spec.maximum, action_dim)
        self.q_update_interval = q_update_interval
        self.buffer_length = buffer_length
        self.replay_buffer = deque([], buffer_length)

        self.observation_dim = sum([i.shape[0] for i in observation_spec.values()])  # number of values in the oberservation space
        self.q = self.make_model()
        self.q_target = self.make_model()
        self.q_target.set_weights(self.q.get_weights())
        self.update_counter = 0  # number of q updates since last q_frozen update (expressed batches)
        self.total_updates = 0

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.q.compile(loss='mse', optimizer=optimizer)

    def select_action(self, time_step, epsilon=.1):
        """ Epsilon greedy action selection """

        if np.random.uniform() < epsilon:
            action_idx = np.random.randint(0, self.action_dim)
        else:
            observation = self.get_observation_vector(time_step)  # concat all observations
            prediction = self.predict(observation[np.newaxis,:], self.q)[0]
            action_idx = np.argmax(prediction)

        return self.action_from_index(action_idx)

    def add_experience(self, time_step, action, time_step_next):
        """ add to replay buffer an experience of form: (observation, action, reward, observation_next, done) """

        self.replay_buffer.append([
            self.get_observation_vector(time_step),
            action,
            time_step_next.reward,
            self.get_observation_vector(time_step_next),
            time_step_next.last()
        ])

    def update(self, batch_size=32, gamma=1):
        """ Update Q function(s) """
        if len(self.replay_buffer) >= batch_size:

            # get batch
            batch = random.sample(self.replay_buffer, batch_size)
            observations = np.array([i[0] for i in batch])
            actions = np.array([i[1] for i in batch])
            a_idx = np.array(self.index_from_action(actions))  # indices for actions
            rewards = np.array([i[2] for i in batch])
            observations_next = np.array([i[3] for i in batch])
            done = np.array([i[4] for i in batch], dtype='bool')

            # stack and predict observations and observations_next at once to increase speed
            stacked = np.vstack((observations, observations_next))
            temp = self.predict(stacked, self.q_target)
            targets = temp[:batch_size]
            targets_next = temp[batch_size:]

            # update targets for selected actions
            targets[np.arange(batch_size), a_idx] = rewards
            targets[np.arange(batch_size)[~done], a_idx[~done]] += gamma * np.max(targets_next[~done], axis=1)

            # update q
            self.q.fit(observations, targets, verbose=False)

            # update q_target if enough updates
            self.total_updates += 1
            self.update_counter += 1
            if self.update_counter == self.q_update_interval:
                self.q_target.set_weights(self.q.get_weights())
                self.update_counter = 0

    def make_model(self, units_per_layer=(24,48)):
        """ Make Q function MLP with softmax output over discrete actions """
        # todo: make work when only a single hidden layer is requested
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units_per_layer[0], activation='tanh', input_dim=self.observation_dim))
        for i in units_per_layer[1:]:
            model.add(tf.keras.layers.Dense(i, activation='tanh'))
        model.add(tf.keras.layers.Dense(self.action_dim, activation='linear'))
        return model

    def index_from_action(self, action):
        """ Converts action(s) in [action_spec.minimum, action_spec.maximum] to integer index in [0,action_dim] """
        # todo: there should be a fast one-liner for the following // rounding errors may affect us here too... // enum?
        if type(action) is float:
            action = [action]
        return [int(np.where(self.actions==a)[0]) for a in action]

    def action_from_index(self, action_idx):
        """ Converts integer index(es) in [0,action_dim] to action in [action_spec.minimum, action_spec.maximum] """
        return self.actions[np.array(action_idx)]

    @staticmethod
    def get_observation_vector(time_step):
        """ converts 'dm_env._environment.TimeStep' to observation vector """
        return np.hstack([v for v in time_step.observation.values()])

    @staticmethod
    def predict(x, model, fast_predict=True):
        """ q_target prediction. fast_predict uses numpy for prediction, which is faster for small networks """
        if fast_predict:
            weights = [layer.get_weights()[0] for layer in model.layers]
            biases = [layer.get_weights()[1] for layer in model.layers]
            y = x.copy()
            for w, b in zip(weights[:-1], biases[:-1]):
                y = np.tanh(np.matmul(y, w) + b)
            y = np.matmul(y, weights[-1]) + biases[-1]
        else:
            y = model.predict(x)
        return y

    def get_prediction_grid(self, bins=10, percentile_lims=(1,99)):
        """ Get hyperrectangle of predictions spanning observation space """
        observations = np.array([i[0] for i in self.replay_buffer])
        axis_limits = np.percentile(observations, percentile_lims, axis=0)
        axis_grids = [np.linspace(dmin, dmax, num=bins) for dmin, dmax in zip(axis_limits[0], axis_limits[1])]
        grid = np.array(np.meshgrid(*axis_grids)).reshape(5, -1).transpose()
        predictions = self.q.predict(grid).reshape([bins] * self.observation_dim + [2])
        return predictions, axis_limits




