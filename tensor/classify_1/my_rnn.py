#!/usr/bin/env python3
#coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)   # set random seed

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001                  # learning rate
training_iters = 100000     # train step 上限
batch_size = 128
n_inputs = 28               # MNIST data input (img shape: 28*28)
n_steps = 28                # time steps
n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              # MNIST classes (0-9 digits)

# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units]),name="w"),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]),name="w")
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]),name="b"),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]),name="b")
}

def RNN(x, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)
    x = tf.reshape(x, [-1, n_inputs])
    with tf.name_scope("input"):
        x_in = tf.matmul(x, weights['in']) + biases['in']
        # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
        x_in = tf.reshape(x_in, [-1, n_steps, n_hidden_units])

    # batch_size = x.shape[0]
    # print(x.shape)
    # 使用 basic LSTM Cell.
    with tf.name_scope("lstm"):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # 初始化全零 state
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)

    with tf.name_scope("output"):
        # 1： 直接调用final_state 中的 h_state (final_state[1]) 来进行运算:
        results = tf.matmul(final_state[1], weights['out']) + biases['out']

        # 2： 把 outputs 变成 列表 [(batch, outputs)..] * steps
        # outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
        # results = tf.matmul(outputs[-1], weights['out']) + biases['out']    #选取最后一个 output

    return results

def train():
    logits = RNN(x, weights,biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)



    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for eposi in range(training_iters):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])

            sess.run([optimizer], feed_dict={
                x: batch_xs,
                y: batch_ys
            })
            if eposi % 50 == 0:
                cs = sess.run(cost, feed_dict={
                    x: batch_xs,
                    y: batch_ys
                })
                batch_vali_xs, batch_vali_ys = mnist.validation.next_batch(batch_size)
                batch_vali_xs = batch_vali_xs.reshape([-1, n_steps, n_inputs])
                print('{}: current training cross entropy is {}'.format(eposi, cs))
                correct_rate = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1)), tf.float32))
                print('{}: current validation correct rate is {}'.format(eposi, sess.run(correct_rate, feed_dict={
                    x: batch_vali_xs,
                    y: batch_vali_ys
                })))
        saver.save(sess, "my_rnn/model_1.ckpt")

def test():
    logits = RNN(x, weights, biases)



    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 提取变量
        saver.restore(sess, "my_rnn/model_1.ckpt")
        batch_test_xs, batch_test_ys = mnist.test.next_batch(batch_size)
        batch_test_xs = batch_test_xs.reshape([-1, n_steps, n_inputs])
        correct_rate = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1)), tf.float32))
        print('test correct rate is {}'.format(sess.run(correct_rate, feed_dict={
            x: batch_test_xs,
            y: batch_test_ys
        })))

# 图片已经被归一化了
# image = mnist.validation.images[0]
# image[image>0]=1
# print(image)

if __name__ == "__main__":
    # training
    # train()
    # test
    test()