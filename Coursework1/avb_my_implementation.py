import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data


# ## Parameters

# In[2]:

params = {
    'batch_size': 512,
    'latent_dim': 2, # dimensionality of latent space
    'eps_dim': 4, # dimensionality of epsilon, used in inference net, z_phi(x, eps)
    'input_dim': 4, # dimensionality of input (also the number of unique datapoints)
    'n_layer_disc': 2, # number of hidden layers in discriminator
    'n_hidden_disc': 256, # number of hidden units in discriminator
    'n_layer_gen': 2,
    'n_hidden_gen': 256,
    'n_layer_inf': 2,
    'n_hidden_inf': 256,
}
points_per_class = params['batch_size'] / params['input_dim']
labels = np.concatenate([[i] * points_per_class for i in xrange(params['input_dim'])])
np_data = np.eye(params['input_dim'], dtype=np.float32)[labels]

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig

def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return st.StochasticTensor(
        ds.MultivariateNormalDiag(mu=tf.zeros(shape), diag_stdev=tf.ones(shape), **kwargs))

def generative_network(batch_size, latent_dim, input_dim, n_layer, n_hidden, eps=1e-6):
    with tf.variable_scope("generative"):
        z = standard_normal([batch_size, latent_dim], name="p_z")
        h = slim.repeat(z.value(), n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        # BUG: BernoulliSigmoidP gives NaNs when log_p is large, so we constrain
        # probabilities to be in (eps, 1-eps) and use Bernoulli
        p = eps + (1-2 * eps) * slim.fully_connected(h, input_dim, activation_fn=tf.nn.sigmoid)
        x = st.StochasticTensor(ds.Bernoulli(p=p, name="p_x"))
    return [x, z]

def inference_network(x, latent_dim, n_layer, n_hidden, eps_dim):
    eps = standard_normal([x.get_shape().as_list()[0], eps_dim], name="eps").value()
    h = tf.concat([x, eps], 1)
    with tf.variable_scope("inference"):
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        z = slim.fully_connected(h, latent_dim, activation_fn=None, scope="q_z")
    return z

def data_network(x, z, n_layers=2, n_hidden=256, activation_fn=None):
    """Approximate log data density."""
    h = tf.concat([x, z], 1)
    with tf.variable_scope('discriminator'):
        h = slim.repeat(h, n_layers, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=activation_fn)
    return tf.squeeze(log_d, squeeze_dims=[1])

tf.reset_default_graph()

x = tf.constant(np_data)
p_x, p_z = generative_network(params['batch_size'], params['latent_dim'], params['input_dim'],
                              params['n_layer_gen'], params['n_hidden_gen'])
q_z = inference_network(x, params['latent_dim'], params['n_layer_inf'], params['n_hidden_inf'],
                       params['eps_dim'])

# Discriminator classifies between (x, z_prior) and (x, z_posterior)
# where z_prior ~ p(z), and zq_z_posterior = q(z, eps) with eps ~ N(0, I)
log_d_prior = data_network(x, p_z.value(), n_layers=params['n_layer_disc'],
                           n_hidden=params['n_hidden_disc'])
log_d_posterior = graph_replace(log_d_prior, {p_z.value(): q_z})
disc_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=log_d_posterior, logits=tf.ones_like(log_d_posterior)) +
    tf.nn.sigmoid_cross_entropy_with_logits(labels=log_d_prior, logits=tf.zeros_like(log_d_prior)))

# Compute log p(x|z) with z ~ p(z), used as a placeholder
recon_likelihood_prior = p_x.distribution.log_prob(x)
# Compute log p(x|z) with z = q(x, eps)
# This is the same as the above expression, but with z replaced by a sample from q instead of p
recon_likelihood = tf.reduce_sum(graph_replace(recon_likelihood_prior, {p_z.value(): q_z}), [1])

# Generator tries to maximize reconstruction log-likelihood while minimizing the discriminator output
gen_loss = tf.reduce_mean(log_d_posterior) - tf.reduce_mean(recon_likelihood)

qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference")
pvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative")
dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
opt = tf.train.AdamOptimizer(1e-3, beta1=0.5)#, epsilon=1e-3)
train_gen_op =  opt.minimize(gen_loss, var_list=qvars + pvars)
train_disc_op = opt.minimize(disc_loss, var_list=dvars)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

it = 0
fs = []
for i in range(10000):
    f, _, _ = sess.run([[gen_loss, disc_loss, log_d_posterior], train_gen_op, train_disc_op])
    print(f)
    fs.append(f)
    if it % 1000 == 0:
        print(f)
        it +=1
        '''reconstructed = sess.run(recon, feed_dict={x: test_xs})

        fig = plot(reconstructed[1:5])
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

        n_examples = 4
        fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
        for example_i in range(n_examples):
            axs[0][example_i].imshow(
                np.reshape(test_xs[example_i, :], (28, 28)))
            axs[1][example_i].imshow(
                np.reshape([reconstructed[example_i, :]], (28, 28)))
        plt.savefig('out2/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        plt.close(fig) '''