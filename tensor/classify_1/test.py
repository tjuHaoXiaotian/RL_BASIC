#!/usr/bin/env python3
#coding=utf-8

import tensorflow as tf

x = tf.placeholder(tf.float32, [None,3], name='x')
y = tf.placeholder(tf.float32, [None,3], name='y')
equals = tf.equal(tf.argmax(x, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))

with tf.Session() as sess:
    print(sess.run(equals, feed_dict={x:[[1,2,3],[4,5,6]], y:[[3,4,5],[6,7,8]]}))
    print(sess.run(accuracy, feed_dict={x:[[1,2,3],[4,5,6]], y:[[3,4,5],[6,7,8]]}))
