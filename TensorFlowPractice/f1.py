from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
layers = tf.keras.layers

def normalize(vector):
    max=vector.max()
    min=vector.min()
    vector = (vector-min)/(max-min)
    return vector

memory = np.load('game_memory.npy')

memory[:, 0] = memory[:, 0]/1200
memory[:, 1] = memory[:, 1]/700
memory[:,2]=normalize(memory[:,2])


# input shape(1,3) observation : memory[:3]
# output shape(1,4) action :

model_eval = tf.keras.Sequential()
model_eval.add(layers.Dense(3, activation='relu'))

for i in range(5):
    model_eval.add(layers.Dense(10, activation='relu'))
model_eval.add(layers.Dense(4, activation='linear'))

model_eval.compile(optimizer=tf.train.AdamOptimizer(0.01),
                   loss='mse',  # mean squared error
                   metrics=['mae'])  # mean absolute error
q_eval=model_eval.predict(memory[:,:3])
q_target=q_eval.copy()
q_target[np.arange(4000),memory[:,3].astype(int)]=memory[:,4]
# model_eval.fit(memory[:,:3],q_target,batch_size=5,epochs=5)

while 1:

    model_eval.fit(memory[:, :3], q_target, batch_size=5, epochs=1)
    plt.clf()
    plt.close('all')
    reward_predict = model_eval.predict(memory[:, :3])
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(memory[:, 0], memory[:, 1], memory[:, 4])
    ax.scatter(memory[:, 0], memory[:, 1], reward_predict[np.arange(4000),memory[:,3].astype(int)])
    ax.set_ylim3d(1, 0)
    ax.set_xlim3d(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    plt.pause(1)
