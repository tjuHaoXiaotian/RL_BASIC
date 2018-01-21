#!/usr/bin/env python3
#coding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    # define layer name
    with tf.name_scope("layer"):
        # define weights name
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([in_size, out_size]))
        # define biase
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([out_size]) + 0.1, name="biases")
        # define lgits
        with tf.name_scope('w_plus_b'):
            out = tf.matmul(inputs, weights) + biases
        if activation_function is None:
            return out
        else:
            return activation_function(out)

def build_data():
    x_data = np.linspace(-1, 1, 256, dtype=np.float32)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    y_data = np.square(x_data) - 0.5 + noise
    return x_data, y_data

def build_network():
    EPISODES = 5000
    data_x, data_y = build_data()
    # 绘图 ==================================
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data_x, data_y, )
    plt.ion()
    plt.show()
    lines = ax.plot(data_x, data_y, 'r-', lw=3)
    # 绘图 ==================================

    with tf.name_scope("inputs"):
        x = tf.placeholder(tf.float32, [None, 1], name="input_x")
        label_y = tf.placeholder(tf.float32, [None, 1], name="label_y")
    layer1 = add_layer(x, 1, 10, activation_function=tf.nn.relu)
    predict = add_layer(layer1, 10, 1, activation_function=None)

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.square(label_y - predict))
    with tf.name_scope("optimizer"):
        # optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss=loss)
        optimizer = tf.train.AdamOptimizer().minimize(loss=loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(init)
        for i in range(EPISODES):
            sess.run(optimizer, feed_dict={x: data_x, label_y:data_y})
            if i % 50 == 0:
                try:
                    pass
                    # ax.lines.remove(lines[0])
                except Exception:
                    pass
                # 绘图 ==================================
                predictions = sess.run(predict, feed_dict={x: data_x})
                lines[0].set_ydata(predictions)
                # lines = ax.plot(data_x, predictions, 'r-', lw=5)
                plt.pause(0.1)
                # 绘图 ==================================
                print(sess.run(loss, feed_dict={x: data_x, label_y:data_y}))
        # 绘图 ==================================
        plt.ioff()
        plt.show()
        # 绘图 ==================================
if __name__ == "__main__":
    build_network()
