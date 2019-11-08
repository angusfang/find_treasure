import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

Input = tf.keras.layers.Input
Dense = tf.keras.layers.Dense
Model = tf.keras.Model

model = tf.keras.models.load_model(
    'model_save_test'
)
model2 = tf.keras.models.clone_model(
    model
)

xs = np.linspace(-1, 1, 1000)
ys = xs ** 3

xs_test = np.linspace(-1, 1, 1000)

model.compile(optimizer=tf.train.AdamOptimizer(0.1),
              loss=tf.losses.mean_squared_error)
model2.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss=tf.losses.mean_squared_error)

iter_train = 0
while 1:
    iter_train = iter_train + 1
    model.fit(xs, ys, batch_size=5, epochs=2)
    model2.fit(xs, ys, batch_size=5, epochs=2)

    plt.clf()
    plt.close('all')
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.scatter(xs, ys)
    ax1.scatter(xs_test, model.predict(xs_test))
    ax2 = fig.add_subplot(212)
    ax2.scatter(xs, ys)
    ax2.scatter(xs_test, model2.predict(xs_test))
    plt.show()
    plt.pause(2)

    # if iter_train > 1:
    #     tf.keras.models.save_model(
    #         model,
    #         'model_save_test'
    #     )
    #     break
