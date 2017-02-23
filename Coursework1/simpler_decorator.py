# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import functools
import tensorflow as tf


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
        function
    return wrapper


class Model:

    def __init__(self, data, target):
        self.data = tf.reshape(data, [-1, 784])
        self.target = target
        self.property
        self.optimize
        self.error

    @lazy_property
    def prediction(self, input):
        data_size = int(self.data.get_shape()[1])
        target_size = int(self.target.get_shape()[1])
        weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
        bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
        incoming = tf.matmul(self.data, weight) + bias
        return tf.nn.softmax(incoming)

    @lazy_property
    def optimize(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        optimizer = tf.train.RMSPropOptimizer(0.005)
        return optimizer.minimize(cross_entropy)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))


if __name__ == '__main__':
    data = tf.placeholder(tf.float32, [None, 28, 28])
    target = tf.placeholder(tf.float32, [None, 10])
    model = Model(data, target)
    session = tf.Session()
    session.run(tf.initialize_all_variables())
'''    train, test = sets.Mnist()()
    for epoch in range(10):
        for _ in range(100):
            batch = train.random_batch(10)
            session.run(model.optimize, {data: batch[0], target: batch[1]})
        error = session.run(model.error, {data: test.data, target: test.target})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))'''