import random
import tensorflow as tf
import numpy as np
from collections import deque
import ipdb

# todo: should be doing several batches with each update?


class Agent:

    def __init__(self, observation_spec, action_spec, action_dim=2, q_update_interval=1000, buffer_length=100000):
        # todo: observation_dim may fail for higher dimensional spaces
        # todo: is deepcopy necessary for q_frozen?

        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.action_dim = action_dim
        self.actions = np.linspace(action_spec.minimum, action_spec.maximum, action_dim)
        self.q_update_interval = q_update_interval
        self.replay_buffer = deque([], buffer_length)

        self.observation_dim = sum([i.shape[0] for i in observation_spec.values()])  # number of values in the oberservation space
        self.q = self.make_model()
        self.q_target = self.make_model()
        self.q_target.set_weights(self.q.get_weights())
        self.update_counter = 0  # number of q updates since last q_frozen update (expressed in experiences, not batches)

    def select_action(self, time_step, epsilon=.1):
        """ Epsilon greedy action selection """
        # todo: continuous output by taking expectation over predictions, and taking continuous random number

        if np.random.uniform() < epsilon:
            action_idx = np.random.randint(0, self.action_dim)
        else:
            observation = self.get_observation_vector(time_step)  # concat all observations
            prediction = self.q_target.predict(np.array([observation]))[0]
            action_idx = np.argmax(prediction)

        action = self.action_from_index(action_idx)

        return action

    def add_experience(self, time_step, action, time_step_next):
        """ add to replay buffer an experience of form: (observation, action, reward, observation_next, done) """

        # only append if time_step is not last, because time_step_next will be post-reset otherwise
        if not time_step.last():
            self.replay_buffer.append([
                self.get_observation_vector(time_step),
                action,
                time_step_next.reward,
                self.get_observation_vector(time_step_next)
            ])

    def update(self, batch_size=32, gamma=.99):
        """ Update Q function(s) """

        # collect data
        if len(self.replay_buffer) >= batch_size:

            # get batch
            batch = random.sample(self.replay_buffer, batch_size)
            observations = np.array([i[0] for i in batch])
            actions = np.array([i[1] for i in batch])
            a_idx = self.index_from_action(actions)  # indices for actions
            rewards = np.array([i[2] for i in batch])
            observations_next = np.array([i[3] for i in batch])

            targets = self.q_target.predict(observations)
            targets_next = self.q_target.predict(observations_next)
            targets[np.arange(batch_size), a_idx] = rewards + gamma * np.max(targets_next, axis=1)

            # update q
            self.q.fit(observations, targets)

            # update q_target if enough updates
            self.update_counter += batch_size
            if self.update_counter > self.q_update_interval:
                self.q_target.set_weights(self.q.get_weights())
                self.update_counter = 0

    def make_model(self, units_per_layer=(32,64), activation='tanh'):
        """ Make Q function MLP with softmax output over discrete actions """
        # todo: optimizer and learning rate...

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units_per_layer[0], activation=activation, input_dim=self.observation_dim))
        for i in units_per_layer[1:]:
            model.add(tf.keras.layers.Dense(i, activation=activation))
        model.add(tf.keras.layers.Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer='adam')

        return model

    @staticmethod
    def get_observation_vector(time_step):
        """ converts 'dm_env._environment.TimeStep' to observation vector """
        return np.hstack([v for v in time_step.observation.values()])

    def action_discrete_to_continuous(self, action):
        """ Converts integer action in [0,action_dim] to action in [action_spec.minimum, action_spec.maximum] """
        action = (action / self.action_dim) * \
                 (self.action_spec.maximum - self.action_spec.minimum) + self.action_spec.minimum
        return action

    def index_from_action(self, action):
        """ Converts action(s) in [action_spec.minimum, action_spec.maximum] to integer index in [0,action_dim] """
        # todo: there should be a fast one-liner for the following // rounding errors may affect us here too...
        if type(action) is float:
            action = [action]
        return [int(np.where(self.actions==a)[0]) for a in action]

    def action_from_index(self, action_idx):
        """ Converts integer index(es) in [0,action_dim] to action in [action_spec.minimum, action_spec.maximum] """
        return self.actions[np.array(action_idx)]

