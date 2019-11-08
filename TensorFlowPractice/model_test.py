import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Model = tf.keras.Model

a = Input(shape=(1,))
for i in range(5):
    b = Dense(50, activation=tf.nn.relu)(a)
b = Dense(1)(b)

model = Model(inputs=a, outputs=b)


xs = np.linspace(-50, 50, 1000)
ys = xs**3

xs_test = np.linspace(-100, 100, 1000)

model.compile(optimizer=tf.train.AdamOptimizer(0.1),
              loss=tf.losses.mean_squared_error)

iter_train = 0
while 1:
    iter_train = iter_train + 1
    model.fit(xs, ys, batch_size=5, epochs=2)

    plt.clf()
    plt.close('all')

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys)
    ax.scatter(xs_test, model.predict(xs_test))
    plt.show()
    plt.pause(2)

    # if iter_train > 1:
    #     tf.keras.models.save_model(
    #         model,
    #         'model_save_test'
    #     )
    #     break
