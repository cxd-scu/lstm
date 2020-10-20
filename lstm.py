import tensorflow as tf
from sklearn.preprocessing import StandardScaler  # 标准化
from load import next_batch, loaddataset


learning_rate = 0.001  # 学习率
n_steps = 76  # LSTM 展开步数（时序持续长度）
n_inputs = 1  # 输入节点数
n_hiddens = 64  # 隐层节点数
n_layers = 2  # LSTM layer 层数
n_classes = 2  # 输出节点数（分类数目）
batch_size = 128


def RNN_LSTM(x, Weights, biases):
    # RNN 输入 reshape
    x = tf.reshape(x, [-1, n_steps, n_inputs])

    # 定义 LSTM cell
    # cell 中的 dropout
    def attn_cell():
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hiddens)
        with tf.name_scope('lstm_dropout'):
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

    # attn_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    # 实现多层 LSTM
    # [attn_cell() for _ in range(n_layers)]
    enc_cells = []
    for i in range(0, n_layers):
        enc_cells.append(attn_cell())
    with tf.name_scope('lstm_cells_layers'):
        mlstm_cell = tf.contrib.rnn.MultiRNNCell(enc_cells, state_is_tuple=True)
    # 全零初始化 state
    _init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    # dynamic_rnn 运行网络
    outputs, states = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state=_init_state, dtype=tf.float32, time_major=False)
    # 输出
    # return tf.matmul(outputs[:,-1,:], Weights) + biases
    return tf.nn.softmax(tf.matmul(outputs[:, -1, :], Weights) + biases)


with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, n_steps * n_inputs], name='x_input')  # 输入
    y = tf.placeholder(tf.float32, [None, n_classes], name='y_input')  # 输出
    keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')  # 保持多少不被 dropout
    batch_size = tf.placeholder(tf.int32, [], name='batch_size_input')  # 批大小
# weights and biases
with tf.name_scope('weights'):
    Weights = tf.Variable(tf.truncated_normal([n_hiddens, n_classes], stddev=0.1),
                          dtype=tf.float32, name='W')
    tf.summary.histogram('output_layer_weights', Weights)
with tf.name_scope('biases'):
    biases = tf.Variable(tf.random_normal([n_classes]), name='b')
    tf.summary.histogram('output_layer_biases', biases)

with tf.name_scope('output_layer'):
    pred = RNN_LSTM(x, Weights, biases)
    tf.summary.histogram('outputs', pred)

# cost
with tf.name_scope('loss'):
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
    tf.summary.scalar('loss', cost)
# optimizer
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuarcy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.name_scope('accuracy'):
    accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=tf.argmax(pred, axis=1))[1]
    tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init)
    data, xtest, label, ytest = loaddataset('D:/source/cicids2018/fri.csv', 1000000)
    data = StandardScaler().fit_transform(data)
    xtest = StandardScaler().fit_transform(xtest)
    test_x, test_y = next_batch(xtest, ytest, 50000, 1)
    train_writer = tf.summary.FileWriter("D:/logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("D:/logs/test", sess.graph)
    step = 1
    for i in range(5000):
        _batch_size = 128
        print(i)
        batch_x, batch_y = next_batch(data, label, _batch_size, i)
        sess.run(train_op, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5, batch_size: _batch_size})
        if (i + 1) % 100 == 0:
            # loss = sess.run(cost, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0, batch_size:_batch_size})
            # acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0, batch_size:_batch_size})
            # print('Iter: %d' % ((i+1) * _batch_size), '| train loss: %.6f' % loss, '| train accuracy: %.6f' % acc)
            train_result = sess.run(merged, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, batch_size: _batch_size})

            test_result = sess.run(merged, feed_dict={x: test_x, y: test_y, keep_prob: 1.0, batch_size: 50000})
            train_writer.add_summary(train_result, i + 1)
            test_writer.add_summary(test_result, i + 1)
        # step += 1

    print("Optimization Finished!")
    # prediction
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.0, batch_size: 50000}))
