"""
test forward pass prediction speed in tensorflow
compare to manual compuation in numpy
"""

import time
import tensorflow as tf
import numpy as np

def make_model(units_per_layer=(32, 64, 128, 10000, 10000, 10000), activation='tanh', input_dim=5, action_dim=3):
    """ Make Q function MLP with softmax output over discrete actions """

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units_per_layer[0], activation=activation, input_dim=input_dim))
    for i in units_per_layer[1:]:
        model.add(tf.keras.layers.Dense(i, activation=activation))
    model.add(tf.keras.layers.Dense(action_dim, activation='linear'))
    model.compile(loss='mse', optimizer='adam')

    return model

model = make_model()
def rand_smp():
    s = np.random.rand()*100
    return np.array([[s,s,s,s,s]], dtype='float32')

# t0=time.time(); model.predict(smp); print(time.time()-t0)

##
smp = rand_smp()


def predict_manual(smp):
    weights = [layer.get_weights()[0] for layer in model.layers]
    biases = [layer.get_weights()[1] for layer in model.layers]

    activation = smp.copy()
    for w, b in zip(weights[:-1], biases[:-1]):
        activation = np.tanh(np.matmul(activation, w) + b)
    return np.matmul(activation, weights[-1]) + biases[-1]




print('manual:    {}'.format(predict_manual(smp)))
print('.predict:  {}'.format(model.predict(smp)))