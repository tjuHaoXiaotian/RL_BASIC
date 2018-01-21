#!/usr/bin/env python3
#coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# generate some data
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

def add_layer(
    inputs ,
    in_size,
    out_size,
    n_layer,
    activation_function=None):
    ## add one more layer and return the output of this layer
    layer_name='layer%s' % (n_layer)
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([in_size, out_size]), name="w")
            # ======== tensorboard： weight ===================================
            tf.summary.histogram(layer_name+'/weights', weights)
        with tf.name_scope("biases"):
            # biases = tf.Variable(tf.random_normal([out_size], mean=0., stddev=0.1), name="b")
            biases = tf.Variable(tf.zeros([out_size]) + 0.1, name="b")
            # ======== tensorboard： biases ===================================
            tf.summary.histogram(layer_name+'/biases', biases)
        with tf.name_scope("wx_plus_b"):
            # ======== tensorboard： wx_plus_b ===================================
            wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
        # ======== tensorboard： layer outputs ===================================
        tf.summary.histogram(layer_name+'/outputs',outputs)
    return outputs



EPISODES = 5000
# 绘图 ==================================
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()
lines = ax.plot(x_data, y_data, 'r-', lw=3)
# 绘图 ==================================

with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_in")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_in")
# add hidden layer
layer1 = add_layer(xs, 1, 16, n_layer=1, activation_function=tf.nn.relu)
# add output layer
predict = add_layer(layer1, 16, 1, n_layer=2, activation_function=None)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.square(ys - predict))
    # ======== tensor board: 绘制 loss 变化曲线 ===================================
    tf.summary.scalar('loss', loss)

with tf.name_scope("optimizer"):
    # optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss=loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss=loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    #  tf.merge_all_summaries() 方法会对我们所有的 summaries 合并到一起
    merged = tf.summary.merge_all()
    # 将上面‘绘画’出的图保存到一个目录中，以方便后期在浏览器中可以浏览
    writer = tf.summary.FileWriter("logs/", sess.graph)

    sess.run(init)
    for i in range(EPISODES):
        sess.run(optimizer, feed_dict={xs: x_data, ys:y_data})
        if i % 50 == 0:
            # 记录数据
            rs = sess.run(merged, feed_dict={xs: x_data, ys:y_data})
            writer.add_summary(rs, i)

            try:
                pass
                # ax.lines.remove(lines[0])
            except Exception:
                pass
            # 绘图 ==================================
            predictions = sess.run(predict, feed_dict={xs: x_data})
            lines[0].set_ydata(predictions)
            # lines = ax.plot(data_x, predictions, 'r-', lw=5)
            plt.pause(0.1)
            # 绘图 ==================================
            print(sess.run(loss, feed_dict={xs: x_data, ys:y_data}))
    # 绘图 ==================================
    plt.ioff()
    plt.show()
    # 绘图 ==================================