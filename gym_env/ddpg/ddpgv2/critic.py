import tensorflow as tf
from tensorflow.contrib import layers
import math

class Critic:
    def __init__(self, sess,action_dim,state_dim):

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = 32
        self.GAMMA = 0.9
        self.num_units_l1 = 50
        self.num_units_l2 = 40
        self.learning_rate = 0.001
        self.update_TDnet_rate = 0.2
        self.reg = layers.l2_regularizer(0.006)
        self.init_var = 0.01

        self.state_input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name='state_input')
        self.actor_input = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name='actor_input')

        self.Q_value_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='TD_Q_value_input')
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward')
        self.terminal = tf.placeholder(dtype=tf.bool, shape=[None, 1], name='terminal')

        with tf.variable_scope('critic'):
            self.Q_output, self.Q_net_var_set = self.create_network(trainable=True)
        with tf.variable_scope('critic_T'):
            self.Q_T_output, self.QT_net_var_set = self.create_network(trainable=False)

        self.build_update_graph(rate=self.update_TDnet_rate)
        self.build_td_target_graph()
        self.build_cost_graph()
        self.build_gradient_graph()

        self.add_summary()
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('/home/wd/tf/summary')

    def create_network(self, trainable):
        l1_s_w = tf.get_variable('l1_s_w',
                                 shape=[self.state_dim, self.num_units_l1],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(-1/math.sqrt(self.state_dim), 1/math.sqrt(self.state_dim)),
                                 regularizer=self.reg,
                                 trainable=trainable)

        l1_a_w = tf.get_variable('l1_a_w',
                                 shape=[self.action_dim, self.num_units_l1],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(-1/math.sqrt(self.action_dim), 1/math.sqrt(self.action_dim)),
                                 regularizer=self.reg,
                                 trainable=trainable)

        l1_b = tf.get_variable('l1_b',
                               shape=[self.num_units_l1],
                               dtype=tf.float32,
                               initializer=tf.random_uniform_initializer(-self.init_var, self.init_var),
                               trainable=trainable)

        l2_w = tf.get_variable('l2_w',
                               shape=[self.num_units_l1, self.num_units_l2],
                               dtype=tf.float32,
                               initializer=tf.random_normal_initializer(-1/math.sqrt(self.num_units_l1), 1/math.sqrt(self.num_units_l1)),
                               regularizer=self.reg,
                               trainable=trainable)

        l2_b = tf.get_variable('l2_b',
                               shape=[self.num_units_l2],
                               dtype=tf.float32,
                               initializer=tf.random_uniform_initializer(-self.init_var, self.init_var),
                               trainable=trainable)

        l3_w = tf.get_variable('l3_w',
                               shape=[self.num_units_l2, 1],
                               dtype=tf.float32,
                               initializer=tf.random_normal_initializer(-1/math.sqrt(self.num_units_l2), 1/math.sqrt(self.num_units_l2)),
                               regularizer=self.reg,
                               trainable=trainable)

        l3_b = tf.get_variable('l3_b',
                               shape=[1],
                               dtype=tf.float32,
                               initializer=tf.random_uniform_initializer(-self.init_var, self.init_var),
                               trainable=trainable)

        l1 = tf.nn.tanh(tf.matmul(self.actor_input, l1_a_w) + tf.matmul(self.state_input, l1_s_w) + l1_b)
        l2 = tf.nn.tanh(tf.matmul(l1, l2_w) + l2_b)
        l3 = tf.matmul(l2, l3_w) + l3_b

        return l3,[l1_s_w,l1_a_w,l1_b,l2_w,l2_b,l3_w,l3_b]


    def build_update_graph(self, rate):
        self.update_T_net_compeletely_op_set = [tf.assign(i[1], i[0]) for i in
                                                zip(self.Q_net_var_set, self.QT_net_var_set)]  # QT = Q

        self.update_T_net_op_set = [tf.assign(i[1], i[0] * rate + i[1] * (1 - rate)) for i in
                                    zip(self.Q_net_var_set, self.QT_net_var_set)]  # QT = r*Q + (1-r)*QT

    def build_td_target_graph(self):
        self.td_target = tf.where(self.terminal,
                                  tf.constant(0, dtype=tf.float32, shape=[self.batch_size, 1]),
                                  self.Q_T_output * self.GAMMA) + self.reward

    def build_cost_graph(self):
        self.cost = tf.reduce_mean(tf.square(self.Q_value_input - self.Q_output))

        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def build_gradient_graph(self):
        gradient_temp = tf.gradients(self.Q_output, self.actor_input)
        # output's shape is [1,batch_size,4]????
        self.gradient = tf.reshape(gradient_temp, (self.batch_size, self.action_dim))

    def add_summary(self):
        self.summary_cost = tf.summary.scalar('critic_cost', self.cost)
        self.summary_Q = tf.summary.scalar('critic_Q_value', tf.reduce_mean(self.Q_output))
        self.summary_gradient = tf.summary.scalar('gradient', tf.reduce_mean(self.gradient))

    def operation_get_TDtarget(self, action_next, state_next, reward, terminal):
        return self.sess.run(self.td_target, feed_dict={self.actor_input: action_next,
                                                        self.state_input: state_next,
                                                        self.reward: reward,
                                                        self.terminal: terminal})

    def operation_critic_learn(self, TDtarget, action, state):
        summary_cost, summary_Q, _ = self.sess.run([self.summary_cost, self.summary_Q, self.train],
                                                   feed_dict={self.Q_value_input: TDtarget,
                                                              self.actor_input: action,
                                                              self.state_input: state})
        return summary_cost, summary_Q

    def operation_get_gradient(self, action, state):
        return self.sess.run([self.summary_gradient, self.gradient], feed_dict={self.actor_input: action,
                                                                                self.state_input: state})

    def operation_update_TDnet_compeletely(self):
        self.sess.run(self.update_T_net_compeletely_op_set)

    def operation_update_TDnet(self):
        self.sess.run(self.update_T_net_op_set)




#############test###########
if __name__ == "__main__":
    pass