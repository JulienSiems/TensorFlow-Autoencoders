import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import functools


def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator



class Model:

    def __init__(self, image, dimensions = [784, 500, 200, 64]):
        self.image = image
        self.dimensions = dimensions
        self.prediction
        self.optimize
        self.error

    @define_scope
    def prediction(self):
        current_input = self.image
        # ENCODER
        encoder = []
        for layer_i, n_output in enumerate(self.dimensions[1:]):
            n_input = int(current_input.get_shape()[1])
            W = tf.Variable(xavier_init(n_input, n_output))
            b = tf.Variable(xavier_init(1, n_output))
            encoder.append(W)
            current_input = tf.nn.elu(tf.add(tf.matmul(current_input, W), b),
                                      name='enclayer' + str(layer_i))

        # latent representation
        z = current_input

        encoder.reverse()

        # DECODER
        count = 0
        for W in encoder:
            W_t = tf.transpose(W)
            b = tf.Variable(xavier_init(1, int(W_t.get_shape()[1])))
            current_input = tf.nn.elu(tf.add(tf.matmul(current_input, W_t), b),
                                      name='declayer' + str(count))
            count += 1

        return current_input

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        return optimizer.minimize(self.error)

    @define_scope
    def error(self):
        error = tf.reduce_sum(tf.pow(tf.sub(self.prediction, self.image), 2))
        return error

def main():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    image = tf.placeholder(tf.float32, [None, 784])
    model = Model(image)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(30):
        test_images = mnist.test.images
        test = np.array([img - mean_img for img in test_images])
        error = sess.run(fetches=model.error, feed_dict={image: test})
        print('Test error {:6.2f}'.format(error))
        for batch_i in range(60):
            batch_xs, _ = mnist.train.next_batch(100)
            train = np.array([img-mean_img for img in batch_xs])
            sess.run(fetches=model.optimize, feed_dict={image: train})

    # Plot example reconstructions
    n_examples = 15
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(model.prediction, feed_dict={image: test_xs_norm})
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape([recon[example_i, :]], (28, 28)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()




if __name__ == '__main__':
  main()