"""
compare fit and evaluation speeds for:
- tensorflow (gpu)
- tensorflow (cpu)
- numpy (forward pass only)
"""

"""
speed (ms):
.predict:    40
.fit:        100-200 
"""

import time
import numpy as np



# settings
disable_gpu = True
units_per_layer = (24,48)
batch_size = 1024


# disable gpus
import tensorflow as tf

if disable_gpu:
    print('DISABLING GPUs')
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'


def make_model(units_per_layer=(32,64,128), activation='tanh', input_dim=5, action_dim=3):
    """ Make Q function MLP with softmax output over discrete actions """

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units_per_layer[0], activation=activation, input_dim=input_dim))
    for i in units_per_layer[1:]:
        model.add(tf.keras.layers.Dense(i, activation=activation))
    model.add(tf.keras.layers.Dense(action_dim, activation='linear'))
    model.compile(loss='mse', optimizer='adam')

    return model


def predict_numpy(smp, model):
    weights = [layer.get_weights()[0] for layer in model.layers]
    biases = [layer.get_weights()[1] for layer in model.layers]

    activation = smp.copy()
    for w, b in zip(weights[:-1], biases[:-1]):
        activation = np.tanh(np.matmul(activation, w) + b)
    return np.matmul(activation, weights[-1]) + biases[-1]


# time operations
model = make_model(units_per_layer=units_per_layer)
# model.summary()


t0 = time.time()
model.predict(np.zeros((1,5)))
print('single evaluation: {:.2f}'.format(time.time() - t0))

t0 = time.time()
model.predict(np.zeros((batch_size,5)))
print('batch evaluation: {:.2f}'.format(time.time() - t0))

t0 = time.time()
model.fit(np.zeros((batch_size,5)), np.zeros((batch_size,1)), verbose=False)
print('fit time: {:.2f}'.format(time.time() - t0))

t0 = time.time()
predict_numpy(np.zeros((1,5)), model)
print('single evaluation (numpy): {:.2f}'.format(time.time() - t0))

t0 = time.time()
predict_numpy(np.zeros((batch_size,5)), model)
print('batch evaluation  (numpy): {:.2f}'.format(time.time() - t0))