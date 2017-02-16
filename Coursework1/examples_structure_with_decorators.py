# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import functools
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Model:

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self._prediction = None
        self._optimize = None
        self._error = None

    @property
    def prediction(self):
        if not self._prediction:
            data_size = int(self.data.get_shape()[1])
            target_size = int(self.target.get_shape()[1])
            weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
            bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
            incoming = tf.matmul(self.data, weight) + bias
            self._prediction = tf.nn.softmax(incoming)
        return self._prediction

    @property
    def optimize(self):
        if not self._optimize:
            cross_entropy = -tf.reduce_sum(self.target, tf.log(self.prediction))
            optimizer = tf.train.RMSPropOptimizer(0.03)
            self._optimize = optimizer.minimize(cross_entropy)
        return self._optimize

    @property
    def error(self):
        if not self._error:
            mistakes = tf.not_equal(
                tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
            self._error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        return self._error

def main():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    model = Model(image, label)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for _ in range(10):
      images, labels = mnist.test.images, mnist.test.labels
      error = sess.run(model.error, {image: images, label: labels})
      print('Test error {:6.2f}%'.format(100 * error))
      for _ in range(60):
        images, labels = mnist.train.next_batch(100)
        sess.run(model.optimize, {image: images, label: labels})


if __name__ == '__main__':
  main()