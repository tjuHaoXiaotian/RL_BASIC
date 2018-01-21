#!/usr/bin/env python3
#codiing=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs, in_size, out_size, layer_num, activation_function=None):
    # define layer name
    layer_name = ("layer%s"%layer_num)
    with tf.name_scope(layer_name):
        # define weights name
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([in_size, out_size]), name="w")
        # define biase
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([out_size]) + 0.1, name="b")
        # define lgits
        with tf.name_scope('w_plus_b'):
            out = tf.matmul(inputs, weights) + biases
        if activation_function is None:
            return out
        else:
            return activation_function(out)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, axis=1), tf.argmax(v_ys, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result

with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])
with tf.name_scope('logits'):
    prediction = add_layer(xs, 784, 10, layer_num=1, activation_function=tf.nn.softmax)
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), axis=1))
with tf.name_scope('optimizer'):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

EPISODES = 10000
with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)

    sess.run(tf.global_variables_initializer())
    for i in range(EPISODES):
        batch_xs, batch_ys = mnist.train.next_batch(128)
        sess.run(train_step, feed_dict={xs:batch_xs, ys: batch_ys})
        if i % 50 == 0:
            print("episode %d: accuracy %f" % (i, compute_accuracy(mnist.test.images, mnist.test.labels)))
