import tensorflow as tf
import numpy as np

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)

class MLP(object):
    def __init__(self):
        self.n_input = 784
        self.n_hidden = 100
        self.n_output = 10
        self.transfer = tf.nn.softplus

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input], name='InputData')
        self.y_ = tf.placeholder(tf.float32, [None, self.n_output], name='LabelData')
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']), name='hiddenlayer1')
        self.y = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'], name='output_layer')

        # cost
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits = self.y))
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))

        # accuracy
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("loss", self.cross_entropy)
        tf.summary.scalar("accuracy", self.accuracy)

        merged_summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        summary_writer = tf.summary.FileWriter('/tmp/tensorflow_logs/example', graph=self.sess.graph)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden), name='weights1')
        all_weights['b1'] = tf.Variable(
            tf.zeros([self.n_hidden]), dtype=tf.float32, name='bias1')
        all_weights['w2'] = tf.Variable(
            tf.zeros([self.n_hidden, self.n_output]), dtype=tf.float32, name='weights2')
        all_weights['b2'] = tf.Variable(
            tf.zeros([self.n_output]), dtype=tf.float32, name='bias2')
        return all_weights

    def partial_fit(self, X, y):
        crossentr, opt = self.sess.run((self.cross_entropy, self.optimizer), feed_dict={self.x: X, self.y_: y})
        return crossentr

    def accuracy_f(self, X, y):
        acc = self.sess.run(self.accuracy, feed_dict={self.x: X, self.y_: y})
        return acc



