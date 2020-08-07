import random
import tensorflow as tf
import numpy as np
from collections import deque
import time
import ipdb

"""
speed tests (s):
- select action:        .017
- select action (np):   .001
- fit:                  .019
- predict batch:        .016
- prepare minibatch:    .0005
- step:                 .0002
"""


class Agent:

    def __init__(self, observation_spec, action_spec, action_grid=2, learning_rate=.001, q_update_interval=100,
                 buffer_length=10000, units_per_layer=(24,48), double_dqn=False):

        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.action_grid = action_grid
        self.q_update_interval = q_update_interval
        self.buffer_length = buffer_length
        self.replay_buffer = deque([], buffer_length)
        self.double_dqn = double_dqn

        # define action and observation spaces
        self.observation_dim = sum([i.shape[0] for i in observation_spec.values()])  # number of values in the oberservation space
        self.action_dim = action_spec.shape[0]
        action_grids = [np.linspace(a_min, a_max, action_grid) for a_min, a_max in
                        zip(action_spec.minimum, action_spec.maximum)]
        self.actions = np.array(np.meshgrid(*action_grids)).reshape(self.action_dim,-1).T  # each row is a unique action vector of length action_dim

        # initialize models
        self.q = self.make_model(units_per_layer=units_per_layer)
        self.q_target = self.make_model(units_per_layer=units_per_layer)
        self.q_target.set_weights(self.q.get_weights())
        self.total_updates = 0
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.q.compile(loss='mse', optimizer=optimizer)

    def select_action(self, time_step, epsilon=.1):
        """ Epsilon greedy action selection """
        if np.random.uniform() < epsilon:
            action_idx = np.random.randint(0, self.action_grid*self.action_dim)
        else:
            observation = self.get_observation_vector(time_step)
            prediction = self.predict(observation[np.newaxis,:], self.q)[0]
            action_idx = np.argmax(prediction)

        return self.action_from_index(action_idx)

    def add_experience(self, time_step, action, time_step_next):
        """ Add to replay buffer an experience of form: (observation, action, reward, observation_next, is_last) """
        self.replay_buffer.append((
            self.get_observation_vector(time_step),
            action,
            time_step_next.reward,
            self.get_observation_vector(time_step_next),
            time_step_next.last()
        ))

    def update(self, batch_size=32, gamma=1):
        """ Update Q function(s) """
        if len(self.replay_buffer) >= batch_size:
            # get mini-batch
            batch = random.sample(self.replay_buffer, batch_size)
            observations = np.array([i[0] for i in batch])
            actions = np.array([i[1] for i in batch])
            a_idx = np.array([self.index_from_action(a) for a in actions])  # indices for actions
            rewards = np.array([i[2] for i in batch])
            observations_next = np.array([i[3] for i in batch])
            done = np.array([i[4] for i in batch], dtype='bool')

            # stack and predict observations and observations_next at once to increase speed
            temp = self.predict(np.vstack((observations, observations_next)), self.q_target)
            targets = temp[:batch_size]
            q_target_predictions = temp[batch_size:]

            # update targets for selected actions
            targets[np.arange(batch_size), a_idx] = rewards
            if self.double_dqn:
                a_max = np.argmax(q_target_predictions, axis=1)  # action that maximizes q_target for next state
                q_predictions = self.predict(observations_next, self.q)  # predictions based on q, not q_target
                targets[np.arange(batch_size)[~done], a_idx[~done]] += \
                    gamma * q_predictions[np.arange(batch_size)[~done], a_max[~done]]
            else:
                targets[np.arange(batch_size)[~done], a_idx[~done]] += \
                    gamma * np.max(q_target_predictions[~done], axis=1)

            # update q
            self.q.fit(observations, targets, verbose=False)

            # update q_target if enough updates
            self.total_updates += 1
            if self.total_updates%self.q_update_interval == 0:
                self.q_target.set_weights(self.q.get_weights())

    def make_model(self, units_per_layer=(24,48)):
        """ Make Q function MLP with softmax output over discrete actions """
        model = tf.keras.Sequential()
        units_per_layer = (units_per_layer,) if isinstance(units_per_layer, int) else units_per_layer  # if only one hidden layer requested
        model.add(tf.keras.layers.Dense(units_per_layer[0], activation='tanh', input_dim=self.observation_dim))
        for i in units_per_layer[1:]:
            model.add(tf.keras.layers.Dense(i, activation='tanh'))
        model.add(tf.keras.layers.Dense(self.action_dim*self.action_grid, activation='linear'))
        return model

    def index_from_action(self, action):
        """ Convert action to int index """
        action = [action] if not isinstance(action[0], list) else action
        return int(np.where((self.actions == np.array(action)).all(axis=1))[0])

    def action_from_index(self, action_idx):
        """ Convert int index(es) to action """
        return self.actions[np.array(action_idx)]

    @staticmethod
    def get_observation_vector(time_step):
        """ Convert 'dm_env._environment.TimeStep' to observation vector """
        return np.hstack([v for v in time_step.observation.values()])

    @staticmethod
    def predict(x, model, fast_predict=True):
        """ q network prediction // fast_predict uses numpy for prediction, which is faster for small networks """
        if fast_predict:
            weights = [layer.get_weights()[0] for layer in model.layers]
            biases = [layer.get_weights()[1] for layer in model.layers]
            prediction = x
            for w, b in zip(weights[:-1], biases[:-1]):
                prediction = np.tanh(np.matmul(prediction, w) + b)
            prediction = np.matmul(prediction, weights[-1]) + biases[-1]
        else:
            prediction = model.predict(x)
        return prediction

    def get_prediction_grid(self, bins=10, percentile_lims=(1,99)):
        """ Get hyper-rectangle of predictions spanning observation space """
        observations = np.array([i[0] for i in self.replay_buffer])
        axis_limits = np.percentile(observations, percentile_lims, axis=0)
        axis_grids = [np.linspace(dmin, dmax, num=bins) for dmin, dmax in zip(axis_limits[0], axis_limits[1])]
        grid = np.array(np.meshgrid(*axis_grids)).reshape(self.observation_dim, -1).transpose()
        predictions = self.q.predict(grid).reshape([bins] * self.observation_dim + [self.action_grid*self.action_dim])
        return predictions, axis_limits




