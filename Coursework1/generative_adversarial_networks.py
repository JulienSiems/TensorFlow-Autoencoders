import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import functools

# TO DO: ADD THE REFERENCES
def xavier_init(fan_in, fan_out, constant = 1):
    with tf.name_scope('xavier'):
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

class GAN:
    def __init__(self, image, noise, discr_dimensions = [784, 64], gener_dimensions = [64, 784]):
        self.image = image
        self.noise = noise
        self.discr_dimensions = discr_dimensions
        self.gener_dimensions = gener_dimensions
        self.optimize_generator
        self.error_generator
        self.optimize_discriminator
        self.error_discriminator

    def discriminator(self, input):
        current_input = self.noise
        # DISCRIMINATOR
        with tf.name_scope('Discriminator'):
            for layer_i, n_output in enumerate(self.discr_dimensions):
                with tf.name_scope('generator_layer' + str(layer_i)):
                    n_input = int(current_input.get_shape()[1])
                    W = tf.Variable(xavier_init(n_input, n_output), name = 'weight'+str(layer_i))
                    b = tf.Variable(tf.zeros(shape=(1, n_output)), name = 'bias'+str(layer_i))
                    current_input = tf.nn.elu(tf.add(tf.matmul(current_input, W), b),
                                              name='discriminator' + str(layer_i))
        return current_input

    def generator(self, input):
        current_input = input
        # GENERATOR
        with tf.name_scope('Generator'):
            for layer_i, n_output in enumerate(self.gener_dimensions):
                with tf.name_scope('generator_layer' + str(layer_i)):
                    n_input = int(current_input.get_shape()[1])
                    W = tf.Variable(xavier_init(n_input, n_output), name = 'weight'+str(layer_i))
                    b = tf.Variable(tf.zeros(shape=(1, n_output)), name = 'bias'+str(layer_i))
                    current_input = tf.nn.elu(tf.add(tf.matmul(current_input, W), b),
                                              name='generator' + str(layer_i))
        return current_input


    @define_scope
    def optimize_generator(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        return optimizer.minimize(self.error_generator)

    @define_scope
    def error_generator(self):
        with tf.name_scope('Loss_generator'):
            self.loss_generator = tf.reduce_mean(tf.log(1-self.discriminator(self.generator(self.noise))))
        tf.summary.scalar('error', self.loss_generator)
        return self.loss_generator

    @define_scope
    def optimize_discriminator(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        return optimizer.minimize(self.error_discriminator)

    @define_scope
    def error_discriminator(self):
        with tf.name_scope('loss_discriminator'):
            self.loss_discriminator = tf.reduce_mean(tf.log(self.discriminator(self.image))
                                                 + tf.log(1-self.discriminator(self.generator(self.noise))))
        tf.summary.scalar('error', self.loss_discriminator)
        return self.loss_discriminator

def main():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    image = tf.placeholder(dtype=tf.float32, shape=[None, 784], name = 'image')
    noise = tf.placeholder(dtype=tf.float32, shape=[None, 64], name = 'noise')
    gan = GAN(image, noise)

    merged_summary = tf.summary.merge_all()

    sess = tf.Session()
    logpath = '/tmp/tensorflow_logs/vae/9'
    n_examples = 100
    test_writer = tf.summary.FileWriter(logpath, graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(10):
        for batch_i in range(1):
            noise_samples = np.random.normal(size=(n_examples, 64))
            batch_xs, _ = mnist.train.next_batch(n_examples)
            _, summary = sess.run(fetches=[gan.optimize_discriminator, merged_summary],
                                  feed_dict={image: batch_xs, noise: noise_samples})
            test_writer.add_summary(summary, epoch_i)
        _, summary = sess.run(fetches=[gan.optimize_generator, merged_summary],
                              feed_dict={noise: noise_samples})

    # Plot example reconstructions
    '''n_examples = 15
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(model.prediction, feed_dict={image: test_xs})
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape([recon[example_i, :]], (28, 28)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()

    n_examples = 15
    test_xs = np.random.normal(size=(n_examples, 64))
    recon = sess.run(model.prediction, feed_dict={model.latent: test_xs})
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (8, 8)))
        axs[1][example_i].imshow(
            np.reshape([recon[example_i, :]], (28, 28)))
    fig.show()
    plt.draw()
    plt.savefig('15_examples.png')
    plt.waitforbuttonpress()'''


if __name__ == '__main__':
  main()