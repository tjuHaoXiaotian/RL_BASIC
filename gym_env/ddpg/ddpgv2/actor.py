import tensorflow as tf
from tensorflow.contrib import layers
import math


class Actor:
    def __init__(self,sess,action_dim,state_dim):
        self.sess = sess
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.batch_size = 32
        self.num_units_l1 = 30
        self.num_units_l2 = 20
        self.learning_rate = 0.001
        self.init_var = 0.01
        self.update_TDnet_rate = 0.2
        self.reg = layers.l2_regularizer(0.006)

        self.state_input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name='state_input')
        self.action_gradient_input = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name='actor_input')

        with tf.name_scope('ACTOR_A_O'):
            with tf.variable_scope('ACTOR_A_V'):
                self.action_output = self.create_network(trainable=True)
        with tf.name_scope('ACTOR_AT_O'):
            with tf.variable_scope('ACTOR_AT_V'):
                self.action_T_output = self.create_network(trainable=False)

        self.gather_var()

        self.build_update_graph(self.update_TDnet_rate)
        self.build_cost_graph()

        self.add_aummary()
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('/home/wd/tf/summary1')

    def create_network(self, trainable):
        l1 = tf.layers.dense(inputs=self.state_input,
                             units=self.num_units_l1,
                             activation=tf.nn.tanh,
                             use_bias=True,
                             kernel_initializer=tf.random_normal_initializer
                             (-1/math.sqrt(self.state_dim), 1/math.sqrt(self.state_dim)),
                             bias_initializer=tf.random_uniform_initializer(-self.init_var, self.init_var),
                             trainable=trainable,
                             name='l1',
                             kernel_regularizer=self.reg)

        l2 = tf.layers.dense(inputs=l1,
                             units=self.num_units_l2,
                             activation=tf.nn.tanh,
                             use_bias=True,
                             kernel_initializer=tf.random_normal_initializer
                             (-1/math.sqrt(self.num_units_l1), 1/math.sqrt(self.num_units_l1)),
                             bias_initializer=tf.random_uniform_initializer(-self.init_var, self.init_var),
                             trainable=trainable,
                             name='l2',
                             kernel_regularizer=self.reg)

        action_ouput = tf.layers.dense(inputs=l2,
                                       units=self.action_dim,
                                       activation=tf.nn.tanh,
                                       use_bias=True,
                                       kernel_initializer=tf.random_normal_initializer(-self.num_units_l2, self.num_units_l2),
                                       bias_initializer=tf.random_uniform_initializer(-self.init_var, self.init_var),
                                       trainable=trainable,
                                       name='action_output',
                                       kernel_regularizer=self.reg)

        return action_ouput

    def gather_var(self):
        graph = tf.get_default_graph()

        l1_w = graph.get_tensor_by_name("ACTOR_A_V/l1/kernel:0")
        l1_b = graph.get_tensor_by_name("ACTOR_A_V/l1/bias:0")
        l2_w = graph.get_tensor_by_name("ACTOR_A_V/l2/kernel:0")
        l2_b = graph.get_tensor_by_name("ACTOR_A_V/l2/bias:0")
        output_w = graph.get_tensor_by_name("ACTOR_A_V/action_output/kernel:0")
        output_b = graph.get_tensor_by_name("ACTOR_A_V/action_output/bias:0")

        l1_w_T = graph.get_tensor_by_name("ACTOR_AT_V/l1/kernel:0")
        l1_b_T = graph.get_tensor_by_name("ACTOR_AT_V/l1/bias:0")
        l2_w_T = graph.get_tensor_by_name("ACTOR_AT_V/l2/kernel:0")
        l2_b_T = graph.get_tensor_by_name("ACTOR_AT_V/l2/bias:0")
        output_w_T = graph.get_tensor_by_name("ACTOR_AT_V/action_output/kernel:0")
        output_b_T = graph.get_tensor_by_name("ACTOR_AT_V/action_output/bias:0")

        self.A_net_var_set = [l1_w, l1_b, l2_w, l2_b, output_w, output_b]
        self.AT_net_var_set = [l1_w_T, l1_b_T, l2_w_T, l2_b_T, output_w_T, output_b_T]

    def build_update_graph(self,rate):
        self.update_T_net_compeletely_op_set = [tf.assign(i[1], i[0]) for i in
                                                zip(self.A_net_var_set, self.AT_net_var_set)]  # AT = A

        self.update_T_net_op_set = [tf.assign(i[1], i[0] * rate + i[1] * (1 - rate)) for i in
                                    zip(self.A_net_var_set, self.AT_net_var_set)]  # AT = r*A + (1-r)*AT

    def build_cost_graph(self):
        self.cost = tf.reduce_mean(self.action_gradient_input*self.action_output)#[batch_size,1]*[batch_size,1]

        # must has a negtive learning_rate
        self.train = tf.train.AdamOptimizer(-self.learning_rate).minimize(self.cost)

    def add_aummary(self):
        self.summary_c = tf.summary.scalar('critic_cost',self.cost)


    def operation_get_action_to_environment(self,state):
        return self.sess.run(self.action_output,feed_dict={self.state_input:state})

    def operation_get_action_to_TDtarget(self,state):
        return self.sess.run(self.action_T_output,feed_dict={self.state_input:state})

    def operation_actor_learn(self,gradient,state):
        self.sess.run(self.train,feed_dict={self.action_gradient_input:gradient,self.state_input:state})

    def operation_update_TDnet_compeletely(self):
        self.sess.run(self.update_T_net_compeletely_op_set)

    def operation_update_TDnet(self):
        self.sess.run(self.update_T_net_op_set)