import numpy as np
import csv
import tensorflow as tf

def build_data():
    data = []
    with open('score.csv',encoding='utf-8') as myFile:
        lines=csv.reader(myFile)
        for line in lines:
            data.append(line)
    data = np.array(data)
    # print(data[:, 2:5])
    input = data[1:, 2:5]
    def filer(arr):
        if arr[0] == arr[1] == arr[2] == '':
            return False
        else:
            return True

    def filter_input(arr):
        if arr[2] == '':
            return False
        else:
            return True

    def filter_test(arr):
        if arr[2] == '':
            return True
        else:
            return False

    input = np.array(list(filter(filer, input)))
    test = np.array(list(filter(filter_test, input)))
    test = np.array(test[:,0:2],dtype=np.float32)

    input = np.array(list(filter(filter_input, input)),dtype=np.float32)
    return input, test

LEARNING_RATE = 0.01
def train():
    x, test = build_data()
    # print(x)
    # print(test)
    y = x[:, -1]
    x = x[:,0:2]
    print(len(x)) # 31
    graph = tf.Graph()

    def get_batch(x, y, batch_size = 31):
        batch_num = len(x) // batch_size
        batch_last = len(x) % batch_size
        if batch_last != 0:
            batch_num += 1
        for ii in range(0,batch_num):
            if ii == batch_num - 1:
                yield x[ii * batch_size:], y[ii * batch_size:]
            else:
                yield x[ii * batch_size: (ii + 1) * batch_size],y[ii * batch_size: (ii + 1) * batch_size]
    with graph.as_default():
        tensor_input = tf.placeholder(tf.float32,[None,2], name="input")
        tensor_label = tf.placeholder(tf.float32,[None], name='label')

        hidden_num = 1
        w1 = tf.Variable(tf.truncated_normal([2,hidden_num],stddev=1.), name='w1')
        b1 = tf.Variable(tf.zeros([hidden_num,]), name='b1')
        # w2 = tf.Variable(tf.truncated_normal([hidden_num,1],stddev=1.), name='w2')
        # b2 = tf.Variable(tf.zeros([1,]), name='b2')

        # h_layer = tf.nn.relu(tf.matmul(tensor_input, w1) + b1, name='h1')
        # output = tf.matmul(h_layer, w2) + b2
        output = tf.matmul(tensor_input, w1) + b1
        loss = tf.reduce_mean(tf.square(tensor_label - output))
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
        saver = tf.train.Saver()
    epochs = 5000
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(0, epochs):
            for (x1,y1) in get_batch(x,y):
                # print(len(x1))
                cur_loss, _ = sess.run((loss, optimizer), feed_dict={
                    tensor_input: x1,
                    tensor_label:y1
                })

                print("Epoch: {}/{}".format(i, epochs),
                      "Train loss: {:.3f}".format(cur_loss))
        saver.save(sess, "checkpoints/predict.ckpt")

        result = sess.run(output,feed_dict={
            tensor_input: test
        })
        print(result)
    data = []
    with open('score.csv',encoding='utf-8') as myFile:
        lines=csv.reader(myFile)
        for line in lines:
            data.append(line)
    data = np.array(data)
    idx = 0
    for line in data:
        if line[2] != "" and line[3] != "" and line[4] == "":
            line[4] = str(result[idx][0])
            idx += 1

    print(data)

    with open("result.csv", "w",encoding='utf-8',newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 先写入columns_name
        for line in data:
            writer.writerow(line[:-1])
        # # 写入多行用writerows
        # writer.writerow(data[0, :-1])
        # writer.writerows(data[1:,:-1])


# def predict():
#     graph = tf.Graph()
#     with graph.as_default():
#         saver = tf.train.Saver()
#     with tf.Session(graph=graph) as sess:
#         saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
#         _, test = build_data()
#         sess.run()
if __name__ == "__main__":
    # train()
    state = tf.Variable(0, name="counter")
    init  = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(state))