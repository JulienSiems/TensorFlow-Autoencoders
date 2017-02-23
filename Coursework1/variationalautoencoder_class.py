import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import functools


# References
# https://danijar.com/structuring-your-tensorflow-models/
# https://jmetzen.github.io/2015-11-27/vae.html
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



class VariationalAutoencoder:

    def __init__(self, image, enc_dimensions = [784, 500, 200, 64], dec_dimensions = [64, 200, 500, 784]):
        self.image = image
        self.enc_dimensions = enc_dimensions
        self.dec_dimensions = dec_dimensions
        self.latent = None
        self.prediction
        self.optimize
        self.error

    @define_scope
    def prediction(self):
        current_input = self.image

        # ENCODER
        encoder = []
        with tf.name_scope('Encoder'):
            for layer_i, n_output in enumerate(self.enc_dimensions[1:-1]):
                with tf.name_scope('enc_layer' + str(layer_i)):
                    n_input = int(current_input.get_shape()[1])
                    W = tf.Variable(xavier_init(n_input, n_output), name = 'weight'+str(layer_i))
                    b = tf.Variable(tf.zeros(shape=(1, n_output)), name = 'bias'+str(layer_i))
                    encoder.append(W)
                    current_input = tf.nn.elu(tf.add(tf.matmul(current_input, W), b),
                                              name='enclayer' + str(layer_i))

        # Latent layer
        with tf.name_scope('LatentLayer'):
            n_input = int(current_input.get_shape()[1])

            with tf.name_scope('mu_layer'):
                mu_weight = tf.Variable(xavier_init(n_input, self.enc_dimensions[-1]), name = 'mu_weight')
                mu_bias = tf.Variable(tf.zeros(shape=(1, self.enc_dimensions[-1])), name = 'mu_bias')
                self.mu = tf.add(tf.matmul(current_input, mu_weight), mu_bias, name = 'mu')
                #tf.summary.image('mu', self.mu)

            with tf.name_scope('logvar_layer'):
                logvar_weight = tf.Variable(xavier_init(n_input, self.enc_dimensions[-1]), name = 'logvar_weight')
                logvar_bias = tf.Variable(tf.zeros(shape=(1, self.enc_dimensions[-1])), name = 'logvar_bias')
                self.logvar = tf.add(tf.matmul(current_input, logvar_weight), logvar_bias, name ='logvar')
                #tf.summary.image('logvar', self.logvar)

            eps = tf.random_normal(shape=(1, self.enc_dimensions[-1]), name = 'gaussian_noise')

            self.latent = tf.add(self.mu, tf.mul(tf.sqrt(tf.exp(self.logvar)), eps), name ='hidden_layer')
            current_input = self.latent

        # DECODER
        with tf.name_scope('Decoder'):
            for layer_i, n_output in enumerate(self.dec_dimensions[1:]):
                with tf.name_scope('dec_layer' + str(layer_i)):
                    n_input = int(current_input.get_shape()[1])
                    W = tf.Variable(xavier_init(n_input, n_output), name = 'weight'+str(layer_i))
                    b = tf.Variable(tf.zeros(shape=(1, n_output)), name = 'bias'+str(layer_i))
                    encoder.append(W)
                    current_input = tf.nn.elu(tf.add(tf.matmul(current_input, W), b),
                                              name='decclayer' + str(layer_i))

        return current_input

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        return optimizer.minimize(self.error)

    @define_scope
    def error(self):
        '''# latent loss
        self.latent_loss = -1/2*tf.reduce_sum(1 + tf.log(tf.square(tf.exp(self.logvar))) - tf.square(self.mu) - tf.square(tf.exp(self.logvar)), name = 'latent_loss')
        self.reconstruction_error = tf.reduce_sum(tf.pow(tf.sub(self.prediction, self.image), 2), name = 'reconstruction_loss')/ (28*28)
        loss = tf.reduce_mean(self.latent_loss + self.reconstruction_error)
        tf.summary.scalar('loss', loss)
        return loss'''
        self.reconstr_error = tf.reduce_sum(tf.pow(tf.sub(self.prediction, self.image), 2))
        self.latent_loss = -1/2*tf.reduce_sum(1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar), name = 'latent_loss')
        self.loss = self.reconstr_error + self.latent_loss
        tf.summary.scalar('error', self.loss)
        return self.loss

def main():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    image = tf.placeholder(tf.float32, [None, 784])
    model = VariationalAutoencoder(image)

    merged_summary = tf.summary.merge_all()
    sess = tf.Session()
    logpath = '/tmp/tensorflow_logs/vae/6'
    test_writer = tf.summary.FileWriter(logpath, graph=tf.get_default_graph())
    #train_writer = tf.summary.FileWriter('/train')
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(80):
        test_images = mnist.test.images
        test = np.array([img - mean_img for img in test_images])
        error, summary = sess.run(fetches=[model.error, merged_summary], feed_dict={image: test_images})
        recerror = sess.run(fetches=model.reconstr_error, feed_dict={image:test_images})
        print(recerror)
        test_writer.add_summary(summary, epoch_i)
        print('Test error {:6.2f}'.format(error))
        for batch_i in range(60):
            batch_xs, _ = mnist.train.next_batch(100)
            train = np.array([img-mean_img for img in batch_xs])
            _, summary = sess.run(fetches=[model.optimize, merged_summary], feed_dict={image: batch_xs})
        #train_writer.add_summary(summary, epoch_i)

    # Plot example reconstructions
    n_examples = 15
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
    plt.waitforbuttonpress()


if __name__ == '__main__':
  main()