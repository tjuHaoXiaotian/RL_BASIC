#!/usr/bin/env python3
#coding=utf-8

import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name="w")

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape),name="b")

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME',name="conv")

def max_pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME',name="max_pooling")

def train():
    xs = tf.placeholder(tf.float32,[None, 784])
    ys = tf.placeholder(tf.float32,[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1,28,28,1])

    with tf.name_scope("conv_1"):
        w1 = weight_variable([5,5,1,32])
        print("w1 name: ",w1.name)
        b1 = bias_variable([32])
        print("b1 name: ",b1.name)

        conv1 = tf.nn.relu(conv2d(x_image, w1) + b1)
        pool1 = max_pooling(conv1)

    with tf.name_scope("conv_2"):
        w2 = weight_variable([5,5,32,64])
        print("w2 name: ", w2.name)
        b2 = bias_variable([64])
        print("b2 name: ", b2.name)
        conv2 = tf.nn.relu(conv2d(pool1, w2) + b2)
        pool2 = max_pooling(conv2)

    with tf.name_scope("fully_con"):
        flat = tf.reshape(pool2,[-1, 7 * 7 * 64])
        w_full = weight_variable([7 * 7 * 64, 1024])
        print("w_full name: ", w_full.name)
        b_full = bias_variable([1024])
        print("b_full name: ", b_full.name)
        full = tf.nn.relu(tf.matmul(flat, w_full) + b_full)
        drop_out = tf.nn.dropout(full, keep_prob=keep_prob)

    with tf.name_scope("output"):
        w_full2 = weight_variable([1024, 10])
        print("w_full2 name: ", w_full2.name)
        b_full2 = bias_variable([10])
        print("b_full2 name: ", b_full2.name)


    prediction = tf.nn.softmax(tf.matmul(drop_out, w_full2) + b_full2)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), axis=1))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    eposide = 1000
    batch_vali_xs, batch_vali_ys = mnist.validation.images, mnist.validation.labels

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for eposi in range(eposide):
            batch_xs, batch_ys = mnist.train.next_batch(128)
            sess.run([optimizer], feed_dict={
                xs: batch_xs,
                ys: batch_ys,
                keep_prob:0.8
            })
            if eposi % 50 == 0:
                cs = sess.run(cross_entropy,feed_dict={
                    xs: batch_xs,
                    ys: batch_ys,
                    keep_prob:1.
                })
                print('{}: current training cross entropy is {}'.format(eposi, cs))
                correct_rate = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, axis=1), tf.argmax(ys, axis=1)),tf.float32))
                print('{}: current validation correct rate is {}'.format(eposi,sess.run(correct_rate,feed_dict={
                    xs: batch_vali_xs,
                    keep_prob:1.,
                    ys: batch_vali_ys
                })))
        saver.save(sess, "my_cnn/model_1.ckpt")


def test():
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 28, 28, 1])

    with tf.name_scope("conv_1"):
        w1 = weight_variable([5,5,1,32])
        print("w1 name: ",w1.name)
        b1 = bias_variable([32])
        print("b1 name: ",b1.name)

        conv1 = tf.nn.relu(conv2d(x_image, w1) + b1)
        pool1 = max_pooling(conv1)

    with tf.name_scope("conv_2"):
        w2 = weight_variable([5,5,32,64])
        print("w2 name: ", w2.name)
        b2 = bias_variable([64])
        print("b2 name: ", b2.name)
        conv2 = tf.nn.relu(conv2d(pool1, w2) + b2)
        pool2 = max_pooling(conv2)

    with tf.name_scope("fully_con"):
        flat = tf.reshape(pool2,[-1, 7 * 7 * 64])
        w_full = weight_variable([7 * 7 * 64, 1024])
        print("w_full name: ", w_full.name)
        b_full = bias_variable([1024])
        print("b_full name: ", b_full.name)
        full = tf.nn.relu(tf.matmul(flat, w_full) + b_full)
        drop_out = tf.nn.dropout(full, keep_prob=keep_prob)

    with tf.name_scope("output"):
        w_full2 = weight_variable([1024, 10])
        print("w_full2 name: ", w_full2.name)
        b_full2 = bias_variable([10])
        print("b_full2 name: ", b_full2.name)

    prediction = tf.nn.softmax(tf.matmul(drop_out, w_full2) + b_full2)
    batch_test_xs, batch_test_ys = mnist.test.images, mnist.test.labels

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 提取变量
        saver.restore(sess,"my_cnn/model_1.ckpt")
        correct_rate = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, axis=1), tf.argmax(ys, axis=1)),tf.float32))
        print('test correct rate is {}'.format(sess.run(correct_rate,feed_dict={
            xs: batch_test_xs,
            keep_prob:1.,
            ys: batch_test_ys
        })))

if __name__ == "__main__":
    # training
    # train()
    # test
    test()