"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
# import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.00,
            e_greedy=0.0,
            replace_target_iter=1000,
            memory_size=100000,
            batch_size=5000,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        mode = 'build'
        tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
        input_eval1 = tf.keras.layers.Input(shape=(self.n_features,))
        layer_eval1 = tf.keras.layers.Dense(20, activation=tf.nn.relu)(input_eval1)


        if mode == 'load':
            model_eval1 = tf.keras.models.load_model(
                'action_model_1'
            )
            model_eval2 = tf.keras.models.load_model(
                'action_model_2'
            )
            model_eval3 = tf.keras.models.load_model(
                'action_model_3'
            )
            model_eval4 = tf.keras.models.load_model(
                'action_model_4'
            )

        if mode == 'build':
            for i in range(10):
                layer_eval1 = tf.keras.layers.Dense(40, activation='relu')(layer_eval1)
            layer_eval1 = tf.keras.layers.Dense(1)(layer_eval1)
            model_eval1 = tf.keras.Model(inputs=input_eval1, outputs=layer_eval1)

            model_eval2 = tf.keras.models.clone_model(
                model_eval1
            )
            model_eval3 = tf.keras.models.clone_model(
                model_eval1
            )
            model_eval4 = tf.keras.models.clone_model(
                model_eval1
            )

        tf.keras.models.Model.compile(
            model_eval1,
            optimizer=tf.train.AdamOptimizer(self.lr),
            loss=tf.losses.mean_squared_error
        )
        tf.keras.models.Model.compile(
            model_eval2,
            optimizer=tf.train.AdamOptimizer(self.lr),
            loss=tf.losses.mean_squared_error
        )
        tf.keras.models.Model.compile(
            model_eval3,
            optimizer=tf.train.AdamOptimizer(self.lr),
            loss=tf.losses.mean_squared_error
        )
        tf.keras.models.Model.compile(
            model_eval4,
            optimizer=tf.train.AdamOptimizer(self.lr),
            loss=tf.losses.mean_squared_error
        )

        model_target1 = tf.keras.models.clone_model(
            model_eval1
        )
        model_target2 = tf.keras.models.clone_model(
            model_eval2
        )
        model_target3 = tf.keras.models.clone_model(
            model_eval3
        )
        model_target4 = tf.keras.models.clone_model(
            model_eval4
        )
        self.model_target1 = model_target1
        self.model_target2 = model_target2
        self.model_target3 = model_target3
        self.model_target4 = model_target4

        self.model_eval1 = model_eval1
        self.model_eval2 = model_eval2
        self.model_eval3 = model_eval3
        self.model_eval4 = model_eval4


        

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        # try: reward more bigger memory more stronger
        if r > 0:
            for i in range(int(r / 200)):
                self.memory[index, :] = transition
                print('add bonus ', i, ' memory')

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        # observation[0][2] = observation[0][2]/500
        actions_value_copy = np.array([[0, 0, 0, 0]])
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            p1 = self.model_eval1.predict(observation)
            p2 = self.model_eval2.predict(observation)
            p3 = self.model_eval3.predict(observation)
            p4 = self.model_eval4.predict(observation)
            actions = np.array([p1[0], p2[0], p3[0], p4[0]])
            actions=actions.reshape((1,4))
            action = actions.argmax()
            actions_value_copy = actions
        else:
            action = np.random.randint(0, self.n_actions)
        return action, actions_value_copy

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:

            self.model_target1.set_weights(self.model_eval1.get_weights())
            self.model_target2.set_weights(self.model_eval2.get_weights())
            self.model_target3.set_weights(self.model_eval3.get_weights())
            self.model_target4.set_weights(self.model_eval4.get_weights())

            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_memory[2]=batch_memory[2]/500

        memory_count = batch_memory.shape[0]

        for i in range(memory_count):
            if batch_memory[i][3].astype(int) == 0:
                try:
                    memory1 = np.vstack((memory1, batch_memory[i][np.array([0, 1, 2, 4,5,6,7])]))
                except:
                    memory1 = batch_memory[i][np.array([0, 1, 2, 4,5,6,7])]
            if batch_memory[i][3].astype(int) == 1:
                try:
                    memory2 = np.vstack((memory2, batch_memory[i][np.array([0, 1, 2, 4,5,6,7])]))
                except:
                    memory2 = batch_memory[i][np.array([0, 1, 2, 4,5,6,7])]
            if batch_memory[i][3].astype(int) == 2:
                try:
                    memory3 = np.vstack((memory3, batch_memory[i][np.array([0, 1, 2, 4,5,6,7])]))
                except:
                    memory3 = batch_memory[i][np.array([0, 1, 2, 4,5,6,7])]
            if batch_memory[i][3].astype(int) == 3:
                try:
                    memory4 = np.vstack((memory4, batch_memory[i][np.array([0, 1, 2, 4,5,6,7])]))
                except:
                    memory4 = batch_memory[i][np.array([0, 1, 2, 4,5,6,7])]

        # q_next, q_eval = self.sess.run(
        #     [self.q_next, self.q_eval],
        #     feed_dict={
        #         self.s_: batch_memory[:, -(self.n_features):],  # fixed params
        #         self.s: batch_memory[:, :self.n_features],  # newest params
        #     })
        # 
        # # change q_target w.r.t q_eval's actionp
        # q_target = q_eval.copy()
        # 
        # batch_index = np.arange(self.batch_size, dtype=np.int32)
        # eval_act_index = batch_memory[:, self.n_features].astype(int)
        # reward = batch_memory[:, self.n_features + 1]

        # q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        # _, self.cost = self.sess.run([self._train_op, self.loss],
        #                              feed_dict={self.s: batch_memory[:, :self.n_features],
        #                                         self.q_target: q_target})
        # self.cost_his.append(self.cost)

        # increasing epsilon
        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        # self.learn_step_counter += 1

        def find_max_reward(memory):
            r1 = self.model_target1.predict(memory[:, -3:])
            r2 = self.model_target2.predict(memory[:, -3:])
            r3 = self.model_target3.predict(memory[:, -3:])
            r4 = self.model_target4.predict(memory[:, -3:])
            r = np.hstack((r1,r2,r3,r4))
            max_r = np.max(r,axis=1).reshape((-1,1))
            return max_r

        try:
            memory1
        except:
            pass
        else:
            print('m1-----')
            memory1 = memory1.reshape((-1, 7))
            max_r = find_max_reward(memory1)
            target = memory1[:, 3].reshape((-1,1))+self.gamma*max_r
            target=target.reshape((-1,1))
            self.model_eval1.fit(memory1[:, :3], target, batch_size=10, epochs=1)
        try:
            memory2
        except:
            pass
        else:
            print('m2-----')
            memory2 = memory2.reshape((-1, 7))
            max_r = find_max_reward(memory2)
            target = memory2[:, 3].reshape((-1,1)) + self.gamma * max_r
            target = target.reshape((-1, 1))
            self.model_eval2.fit(memory2[:, :3], target, batch_size=10, epochs=1)
        try:
            memory3
        except:
            pass
        else:
            print('m3-----')
            memory3 = memory3.reshape((-1, 7))
            max_r = find_max_reward(memory3)
            target = memory3[:, 3].reshape((-1,1)) + self.gamma * max_r
            target = target.reshape((-1, 1))
            self.model_eval3.fit(memory3[:, :3], target, batch_size=10, epochs=1)
        try:
            memory4
        except:
            pass
        else:
            print('m4-----')
            memory4 = memory4.reshape((-1, 7))
            max_r = find_max_reward(memory4)
            target = memory4[:, 3].reshape((-1,1)) + self.gamma * max_r
            target = target.reshape((-1, 1))
            self.model_eval4.fit(memory4[:, :3],  target, batch_size=10, epochs=1)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        xs = np.linspace(0, 1, 100)
        ys = np.linspace(0, 1, 100)
        grid = []
        for x in xs:
            for y in ys:
                grid.append(np.array([x, y, 0]))
        grid = np.array(grid)
        reward_predict1 = self.model_eval1.predict(grid[:, :3])
        reward_predict2 = self.model_eval2.predict(grid[:, :3])
        reward_predict3 = self.model_eval3.predict(grid[:, :3])
        reward_predict4 = self.model_eval4.predict(grid[:, :3])


        plt.clf()
        plt.close('all')
        fig = plt.figure()


        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222, projection='3d')
        ax3 = fig.add_subplot(223, projection='3d')
        ax4 = fig.add_subplot(224, projection='3d')
        ax1.scatter(self.memory[:, 0], self.memory[:, 1], self.memory[:, 4])
        ax1.scatter(grid[:, 0], grid[:, 1], reward_predict1)
        ax2.scatter(self.memory[:, 0], self.memory[:, 1], self.memory[:, 4])
        ax2.scatter(grid[:, 0], grid[:, 1], reward_predict2)
        ax3.scatter(self.memory[:, 0], self.memory[:, 1], self.memory[:, 4])
        ax3.scatter(grid[:, 0], grid[:, 1], reward_predict3)
        ax4.scatter(self.memory[:, 0], self.memory[:, 1], self.memory[:, 4])
        ax4.scatter(grid[:, 0], grid[:, 1], reward_predict4)

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



    def output_memory(self):
        np.save('game_memory.npy', self.memory)
        print('save_memory')
        tf.keras.models.save_model(
            self.model_eval1,
            'model_eval1_save'
        )
        tf.keras.models.save_model(
            self.model_eval2,
            'model_eval2_save'
        )
        tf.keras.models.save_model(
            self.model_eval3,
            'model_eval3_save'
        )
        tf.keras.models.save_model(
            self.model_eval4,
            'model_eval4_save'
        )
