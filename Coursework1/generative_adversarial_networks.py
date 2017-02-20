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


class Discriminator:
    def __init__(self, image, generator, discr_dimensions = [784, 64]):
        self.g_sample = generator.predict
        self.image = image
        self.generator = generator
        self.discr_dimensions = discr_dimensions
        self.g_mode = False
        self.predict
        self.optimize
        self.loss

    @define_scope
    def predict(self):
        if self.g_mode is True:
            current_input = self.image
        else:
            current_input = self.g_sample

        # DISCRIMINATOR
        with tf.name_scope('Discriminator'):
            for layer_i, n_output in enumerate(self.discr_dimensions[1:-1]):
                with tf.name_scope('generator_elu_layer' + str(layer_i)):
                    n_input = int(current_input.get_shape()[1])
                    W = tf.Variable(xavier_init(n_input, n_output), name = 'weight'+str(layer_i))
                    b = tf.Variable(tf.zeros(shape=(1, n_output)), name = 'bias'+str(layer_i))
                    current_input = tf.nn.elu(tf.add(tf.matmul(current_input, W), b,
                                                     name='discriminator' + str(layer_i)))

            with tf.name_scope('generator_sig_flat_layer'):
                n_input = int(current_input.get_shape()[1])
                n_output = int(self.discr_dimensions[-1])
                W = tf.Variable(xavier_init(n_input, n_output), name='weight')
                b = tf.Variable(tf.zeros(shape=(1, n_output)), name='bias')
                D = tf.add(tf.matmul(current_input, W), b)
                D_logit = tf.nn.sigmoid(D)
        return D_logit

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(0.001)
        D_loss, D_logit_fake = self.loss
        return optimizer.minimize(D_loss), D_logit_fake

    @define_scope
    def loss(self):
        self.g_sample = self.generator.predict
        D_logit_real = self.predict
        self.g_mode = True
        D_logit_fake = self.predict
        self.g_mode = False

        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_real, tf.ones_like(D_logit_real)))
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_fake, tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + self.D_loss_fake
        return D_loss, D_logit_fake


class Discriminator_fake:
    def __init__(self):
        self.loss

    @define_scope
    def loss(self):
        return tf.Variable(tf.constant(1, shape=[None,1]), dtype=tf.float32),  tf.Variable(tf.constant(1, shape=[None, None]), dtype=tf.float32)



class Generator:
    def __init__(self, noise, discriminator = None, gener_dimensions = [64, 200, 784]):
        self.noise = noise
        self.discriminator = discriminator
        self.D_logit_fake = discriminator.loss[1]
        self.gener_dimensions = gener_dimensions
        self.predict
        #self.optimize
        #self.loss

    @define_scope
    def predict(self):
        current_input = self.noise
        # GENERATOR
        with tf.name_scope('Generator'):
            for layer_i, n_output in enumerate(self.gener_dimensions[1:-1]):
                with tf.name_scope('generator_elu_layer' + str(layer_i)):
                    n_input = int(current_input.get_shape()[1])
                    W = tf.Variable(xavier_init(n_input, n_output), name = 'weight'+str(layer_i))
                    b = tf.Variable(tf.zeros(shape=(1, n_output)), name = 'bias'+str(layer_i))
                    current_input = tf.nn.elu(tf.add(tf.matmul(current_input, W), b,
                                                     name='discriminator' + str(layer_i)))
            with tf.name_scope('generator_sig_layer'):
                n_input = int(current_input.get_shape()[1])
                n_output = int(self.gener_dimensions[-1])
                W = tf.Variable(xavier_init(n_input, n_output), name='weight')
                b = tf.Variable(tf.zeros(shape=(1, n_output)), name='bias')
                current_input = tf.nn.sigmoid(tf.add(tf.matmul(current_input, W), b))
        return current_input

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(0.001)
        return optimizer.minimize(self.loss)

    @define_scope
    def loss(self):
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logit_fake,
                                                                        tf.ones_like(self.D_logit_fake)))
        return G_loss



def main():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)

    image = tf.placeholder(dtype=tf.float32, shape=[None, 784], name = 'image')
    g_sample = tf.placeholder(dtype=tf.float32, shape=[None, 784], name = 'g_sample')
    noise = tf.placeholder(dtype=tf.float32, shape=[None, 64], name = 'noise')
    d_logit_fake = tf.placeholder(dtype=tf.float32, shape=[None, 64], name = 'd_logit_fake')

    discriminator_fake = Discriminator_fake()
    generator = Generator(noise, discriminator_fake)
    discriminator = Discriminator(image, generator)
    generator.discriminator = discriminator

    merged_summary = tf.summary.merge_all()

    sess = tf.Session()
    logpath = '/tmp/tensorflow_logs/gan/1'
    n_examples = 100
    test_writer = tf.summary.FileWriter(logpath, graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(10):
        for batch_i in range(1):
            noise_samples = np.random.normal(size=(n_examples, 64))
            batch_xs, _ = mnist.train.next_batch(n_examples)
            #g_sampled = sess.run(fetches=generator.predict,
                          #        feed_dict={noise: noise_samples})
            sess.run(fetches=discriminator.optimize, feed_dict={image:batch_xs, noise:noise_samples})
            sess.run(fetches=generator.optimize, feed_dict={})
            #sess.run(fetches=generator.optimize, feed_dict={d_logit_fake:D_logit_fake})
            #test_writer.add_summary(summary, epoch_i)
        #_, summary = sess.run(fetches=[gan.optimize_generator, merged_summary],
                              #feed_dict={g_sample: noise_samples})

if __name__ == '__main__':
  main()


  #what is the problem?