import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)

class MLP:
    def __init__(self, X, y):
        self.data = X
        self.target = y
        self._prediction = None
        self._optimize = None
        self._error = None

    def prediction(self):
        if self._prediction is None:
            data_size = int(self.data.get_shape()[1])
            target_size = int(self.target.get_shape()[1])
            weight = tf.Variable(tf.truncated_normal([data_size, target_size]), name = 'weight')
            #bias = tf.Variable(tf.truncated_normal([1, target_size]), name = 'bias')
            incoming = tf.matmul(self.data, weight) #+ bias
            self._prediction = tf.nn.softmax(incoming)
        return self._prediction

    '''def initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden), name='weights1')
        all_weights['b1'] = tf.Variable(
            tf.zeros([self.n_hidden]), dtype=tf.float32, name='bias1')
        all_weights['w2'] = tf.Variable(
            tf.zeros([self.n_hidden, self.n_output]), dtype=tf.float32, name='weights2')
        all_weights['b2'] = tf.Variable(
            tf.zeros([self.n_output]), dtype=tf.float32, name='bias2')
        return all_weights'''

    def optimize(self):
        if self._optimize is None:
            logprob = tf.log(self.prediction + 1e-12)
            cross_entropy = -tf.reduce_sum(self.target * logprob)
            optimizer = tf.train.RMSPropOptimizer(0.03)
            self._optimize = optimizer.minimize(cross_entropy)
        return self._optimize

    def error(self):
        if self._error is None:
            mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
            self._error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        return self._error

def main():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    model = MLP(image, label)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for _ in range(10):
        images, labels = mnist.test.images, mnist.test.labels
        error = sess.run(model.error, {image: images, label: labels})
        print('Test error {:6.2f}%'.format(100 * error))
        sess.run(tf.global_variables_initializer())
        for _ in range(60):
            images, labels = mnist.train.next_batch(100)
            sess.run(model.optimize, {image: images, label: labels})


if __name__ == '__main__':
    main()