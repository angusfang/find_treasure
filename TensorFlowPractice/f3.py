from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

layers = tf.keras.layers
Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Model = tf.keras.Model


def normalize(vector):
    max = vector.max()
    min = vector.min()
    vector = (vector - min) / (max - min)
    return vector


memory = np.load('game_memory.npy')

memory[:, 0] = memory[:, 0] / 1200
memory[:, 1] = memory[:, 1] / 700
memory[:, 2] /= 500

# input shape(1,3) observation : memory[:3]
# output shape(1,4) action :


# model1
input1 = Input(shape=(3,))
layer1 = Dense(20, activation=tf.nn.relu)(input1)
for i in range(10):
    layer1 = Dense(40, activation=tf.nn.relu)(layer1)
layer1 = Dense(1)(layer1)
model_eval1 = Model(inputs=input1, outputs=layer1)

model_eval2 = tf.keras.models.clone_model(
    model_eval1
)
model_eval3 = tf.keras.models.clone_model(
    model_eval1
)
model_eval4 = tf.keras.models.clone_model(
    model_eval1
)


plot_model = tf.keras.utils.plot_model
plot_model(model_eval1, to_file='model.png')

model_eval1.compile(optimizer=tf.train.AdamOptimizer(0.01),
                    loss='mse',  # mean squared error
                    metrics=['mae'])  # mean absolute error

model_eval2.compile(optimizer=tf.train.AdamOptimizer(0.01),
                    loss='mse',  # mean squared error
                    metrics=['mae'])  # mean absolute error

model_eval3.compile(optimizer=tf.train.AdamOptimizer(0.01),
                    loss='mse',  # mean squared error
                    metrics=['mae'])  # mean absolute error

model_eval4.compile(optimizer=tf.train.AdamOptimizer(0.01),
                    loss='mse',  # mean squared error
                    metrics=['mae'])  # mean absolute error

memory_count = memory.shape[0]

for i in range(memory_count):
    if memory[i][3].astype(int) == 0:
        try:
            memory1 = np.vstack((memory1, memory[i][np.array([0,1,2,4])]))
        except:
            memory1 = memory[i][np.array([0,1,2,4])]
    if memory[i][3].astype(int) == 1:
        try:
            memory2 = np.vstack((memory2, memory[i][np.array([0,1,2,4])]))
        except:
            memory2 = memory[i][np.array([0,1,2,4])]
    if memory[i][3].astype(int) == 2:
        try:
            memory3 = np.vstack((memory3, memory[i][np.array([0,1,2,4])]))
        except:
            memory3 = memory[i][np.array([0,1,2,4])]
    if memory[i][3].astype(int) == 3:
        try:
            memory4 = np.vstack((memory4, memory[i][np.array([0,1,2,4])]))
        except:
            memory4 = memory[i][np.array([0,1,2,4])]

# q_eval = model_eval.predict(memory[:, :3])
# q_target = q_eval.copy()
# q_target[np.arange(4000), memory[:, 3].astype(int)] = memory[:, 4]
# model_eval.fit(memory[:,:3],q_target,batch_size=5,epochs=5)

i=0

xs=np.linspace(0,1,100)
ys=np.linspace(0,1,100)
grid=[]
for x in xs:
    for y in ys:
        grid.append(np.array([x,y,0]))
grid=np.array(grid)

while 1:
    i=i+1



    model_eval1.fit(memory1[:, :3], memory1[:, 3], batch_size=20, epochs=1)
    model_eval2.fit(memory2[:, :3], memory2[:, 3], batch_size=20, epochs=1)
    model_eval3.fit(memory3[:, :3], memory3[:, 3], batch_size=20, epochs=1)
    model_eval4.fit(memory4[:, :3], memory4[:, 3], batch_size=20, epochs=1)



    l1=model_eval1.evaluate(memory1[:, :3], memory1[:, 3])
    l2=model_eval2.evaluate(memory2[:, :3], memory2[:, 3])
    l3=model_eval3.evaluate(memory3[:, :3], memory3[:, 3])
    l4=model_eval4.evaluate(memory4[:, :3], memory4[:, 3])
    if (l1[1]+l2[1]+l3[1]+l4[1])<50:
        plt.clf()
        plt.close('all')

        reward_predict1 = model_eval1.predict(grid[:, :3])
        reward_predict2 = model_eval2.predict(grid[:, :3])
        reward_predict3 = model_eval3.predict(grid[:, :3])
        reward_predict4 = model_eval4.predict(grid[:, :3])

        fig = plt.figure()
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        ax3 = fig.add_subplot(223, projection='3d')
        ax4 = fig.add_subplot(224, projection='3d')

        ax1.scatter(grid[:, 0], grid[:, 1], reward_predict1)
        ax2.scatter(grid[:, 0], grid[:, 1], reward_predict2)
        ax3.scatter(grid[:, 0], grid[:, 1], reward_predict3)
        ax4.scatter(grid[:, 0], grid[:, 1], reward_predict4)

        ax1.scatter(memory1[:, 0], memory1[:, 1], memory1[:, 3])
        ax2.scatter(memory2[:, 0], memory2[:, 1], memory2[:, 3])
        ax3.scatter(memory3[:, 0], memory3[:, 1], memory3[:, 3])
        ax4.scatter(memory4[:, 0], memory4[:, 1], memory4[:, 3])


        def set_subplot(ax):
            ax.set_ylim3d(1, 0)
            ax.set_xlim3d(0, 1)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')


        set_subplot(ax1)
        set_subplot(ax2)
        set_subplot(ax3)
        set_subplot(ax4)

        plt.show()

        print('save')
        tf.keras.models.save_model(
            model_eval1,
            './action_model_1',
            overwrite=True,
            include_optimizer=True,

        )
        tf.keras.models.save_model(
            model_eval2,
            './action_model_2',
            overwrite=True,
            include_optimizer=True,

        )
        tf.keras.models.save_model(
            model_eval3,
            './action_model_3',
            overwrite=True,
            include_optimizer=True,

        )
        tf.keras.models.save_model(
            model_eval4,
            './action_model_4',
            overwrite=True,
            include_optimizer=True,

        )
        break
