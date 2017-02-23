import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import functools
slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace


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



class AdversarialVariationalAutoencoder:

    def __init__(self,
                 x,
                 batch_size = 512,

                 latent_dim_generator = 2,
                 input_dim_generator = 4,
                 n_layer_generator = 2,
                 n_hidden_generator = 256,

                 latent_dim_inference = 2,
                 n_layer_inference = 2,
                 n_hidden_inference = 256,
                 eps_dim_inference = 4,


                 n_layers_data=2,
                 n_hidden_data=256,
                 activation_fn_data=None):
        self.x = x
        self.batch_size = batch_size

        self.latent_dim_generator = latent_dim_generator
        self.input_dim_generator = input_dim_generator
        self.n_layer_generator = n_layer_generator
        self.n_hidden_generator = n_hidden_generator

        self.latent_dim_inference = latent_dim_inference
        self.n_layer_inference = n_layer_inference
        self.n_hidden_inference = n_hidden_inference
        self.eps_dim_inference = eps_dim_inference

        self.n_layers_data = n_layers_data
        self.n_hidden_data = n_hidden_data
        self.activation_fn_data = activation_fn_data

        self.px
        self.pz
        self.prediction
        self.optimize
        self.error
        self.generative_network
        self.inference_network
        self.data_network

    def __call__(self, *args, **kwargs):
        tf.reset_default_graph()
        self.px, self.pz = self.generative_network
        q_z = self.inference_network
        log_d_prior = self.data_network
        log_d_posterior = graph_replace(log_d_prior, {self.pz.value(): })


    @define_scope
    def generative_network(self):
        with tf.variable_scope("generative"):
            eps = 1e-6
            self.pz = st.StochasticTensor(ds.MultivariateNormalDiag(mu=tf.zeros([self.batch_size, self.latent_dim_generator]),
                                                              diag_stdev=tf.ones([self.batch_size, self.latent_dim_generator]),
                                                              name="p_z"))
            h = slim.repeat(self.pz.value(), self.n_layer_generator, slim.fully_connected, self.n_hidden_generator, activation_fn=tf.nn.relu)
            # BUG: BernoulliSigmoidP gives NaNs when log_p is large, so we constrain
            # probabilities to be in (eps, 1-eps) and use Bernoulli
            p = eps + (1 - 2 * eps) * slim.fully_connected(h, self.input_dim_generator, activation_fn=tf.nn.sigmoid)
            self.px = st.StochasticTensor(ds.Bernoulli(p=p, name="p_x"))
        return [self.px, self.pz]

    @define_scope
    def inference_network(self):
        eps = st.StochasticTensor(ds.MultivariateNormalDiag(mu=tf.zeros([self.x.get_shape().as_list()[0], self.eps_dim_inference]),
                                      diag_stdev=tf.ones([self.x.get_shape().as_list()[0], self.eps_dim_inference])), name = 'eps').value()
        h = tf.concat_v2([self.x, eps], 1)
        with tf.variable_scope("inference"):
            h = slim.repeat(h, self.n_layer_inference, slim.fully_connected, self.n_hidden_inference, activation_fn=tf.nn.relu)
            z = slim.fully_connected(h, self.latent_dim_inference, activation_fn=None, scope="q_z")
        return z

    @define_scope
    def data_network(self):
        """Approximate log data density."""
        h = tf.concat_v2([self.x, self.pz.value()], 1)
        with tf.variable_scope('discriminator'):
            h = slim.repeat(h, self.n_hidden_data, slim.fully_connected, self.n_hidden_data, activation_fn=tf.nn.relu)
            log_d = slim.fully_connected(h, 1, activation_fn=self.activation_fn_data)
        return tf.squeeze(log_d, squeeze_dims=[1])

    '''
    @define_scope
    def error(self):
        # latent loss
        self.latent_loss = -1/2*tf.reduce_sum(1 + tf.log(tf.square(tf.exp(self.logvar))) - tf.square(self.mu) - tf.square(tf.exp(self.logvar)), name = 'latent_loss')
        self.reconstruction_error = tf.reduce_sum(tf.pow(tf.sub(self.prediction, self.image), 2), name = 'reconstruction_loss')/ (28*28)
        loss = tf.reduce_mean(self.latent_loss + self.reconstruction_error)
        tf.summary.scalar('loss', loss)
        return loss
        self.reconstr_error = tf.reduce_sum(tf.pow(tf.sub(self.prediction, self.image), 2))
        self.latent_loss = -1/2*tf.reduce_sum(1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar), name = 'latent_loss')
        self.loss = self.reconstr_error + self.latent_loss
        tf.summary.scalar('error', self.loss)
        return self.loss
    '''
def main():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    image = tf.placeholder(tf.float32, [None, 784])
    model = AdversarialVariationalAutoencoder(image)

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