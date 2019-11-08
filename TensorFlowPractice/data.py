import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def normalize(vector):
    max = vector.max()
    min = vector.min()
    vector = (vector - min) / (max - min)
    return vector


memory = np.load('game_memory.npy')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# memory[:, 0] = normalize(memory[:, 0])
# memory[:, 1] = normalize(memory[:, 1])
# memory[:, 2] = normalize(memory[:, 2])
memory[:, 0] = memory[:, 0]/1200
memory[:, 1] = memory[:, 1]/700

# ax.scatter(memory[:, 0], memory[:, 1], memory[:, 4])

# norm = normalize(memory[:,:])
# plt.scatter(memory[:,0],memory[:,1],c=norm)


# ax.set_ylim3d(1, 0)
# ax.set_xlim3d(0, 1)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()

# struct model
import tensorflow as tf

layers = tf.keras.layers

model1 = tf.keras.Sequential()
model1.add(layers.Dense(3, activation='relu'))
for i in range(10):
    model1.add(layers.Dense(20, activation='relu'))
model1.add(layers.Dense(4, activation='linear'))

model1.compile(optimizer=tf.train.AdamOptimizer(0.05),
               loss='mse',
               metrics=['mae'])
# model1.compile(optimizer=tf.train.AdamOptimizer,
#                    loss='mse',  # mean squared error
#                    metrics=['mae'])  # mean absolute error
np.random.seed(seed=5)
m1 = np.array(memory[:, 4])
m1=m1.reshape((-1,1))
m2 = np.hstack((m1,m1*0,m1*0+1,m1*3))
while 1:

    model1.fit(memory[:, :3], m2, batch_size=20, epochs=1)
    plt.clf()
    plt.close('all')
    reward_predict = model1.predict(memory[:, :3])
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(memory[:, 0], memory[:, 1], memory[:, 4])
    ax.scatter(memory[:, 0], memory[:, 1], reward_predict[:,3])
    ax.set_ylim3d(1, 0)
    ax.set_xlim3d(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    plt.pause(1)

