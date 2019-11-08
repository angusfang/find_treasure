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
            learning_rate=0.005,
            reward_decay=0.00,
            e_greedy=0.0,
            replace_target_iter=1000,
            memory_size=100000,
            batch_size=1000,
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
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 4, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.tanh(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('lh1'):
                wh1 = tf.get_variable('wh1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                bh1 = tf.get_variable('bh1', [1, n_l1], initializer=b_initializer, collections=c_names)
                lh1 = (tf.matmul(l1, wh1) + bh1)

            with tf.variable_scope('lh2'):
                wh2 = tf.get_variable('wh2', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                bh2 = tf.get_variable('bh2', [1, n_l1], initializer=b_initializer, collections=c_names)
                lh2 = (tf.matmul(lh1, wh2) + bh2)

            with tf.variable_scope('lh3'):
                wh3 = tf.get_variable('wh3', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                bh3 = tf.get_variable('bh3', [1, n_l1], initializer=b_initializer, collections=c_names)
                lh3 = (tf.matmul(lh2, wh3) + bh3)

            with tf.variable_scope('lh4'):
                wh4 = tf.get_variable('wh4', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                bh4 = tf.get_variable('bh4', [1, n_l1], initializer=b_initializer, collections=c_names)
                lh4 = (tf.matmul(lh3, wh4) + bh4)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(lh4, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.tanh(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('lh1'):
                wh1 = tf.get_variable('wh1', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                bh1 = tf.get_variable('bh1', [1, n_l1], initializer=b_initializer, collections=c_names)
                lh1 = (tf.matmul(l1, wh1) + bh1)

            with tf.variable_scope('lh2'):
                wh2 = tf.get_variable('wh2', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                bh2 = tf.get_variable('bh2', [1, n_l1], initializer=b_initializer, collections=c_names)
                lh2 = (tf.matmul(lh1, wh2) + bh2)

            with tf.variable_scope('lh3'):
                wh3 = tf.get_variable('wh3', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                bh3 = tf.get_variable('bh3', [1, n_l1], initializer=b_initializer, collections=c_names)
                lh3 = (tf.matmul(lh2, wh3) + bh3)

            with tf.variable_scope('lh4'):
                wh4 = tf.get_variable('wh4', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
                bh4 = tf.get_variable('bh4', [1, n_l1], initializer=b_initializer, collections=c_names)
                lh4 = (tf.matmul(lh3, wh4) + bh4)


            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(lh4, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        # try: reward more bigger memory more stronger
        if r>0 :
            for i in range(int(r/200)):
                self.memory[index, :] = transition
                print('add bonus ',i,' memory')

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        actions_value_copy=np.array([[0,0,0,0]])
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            actions_value_copy = actions_value
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action,actions_value_copy

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -(self.n_features):],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's actionp
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

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
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def output_memory(self):
        np.save('game_memory.npy',self.memory)
        print('save_memory')


