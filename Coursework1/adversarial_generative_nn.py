import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


# TO DO: ADD THE REFERENCES
def xavier_init(fan_in, fan_out, constant = 1):
    with tf.name_scope('xavier'):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out),
                                 minval = low, maxval = high,
                                 dtype = tf.float32)

noise_dimension = 100
image_dimension = 784
discr_dimensions = [image_dimension, 500, 100, 64]
gener_dimensions = [noise_dimension, 200, 500, 784]

example_image = tf.placeholder(dtype=tf.float32, shape=[None, image_dimension], name='image_dimension')
noise = tf.placeholder(dtype = tf.float32, shape=[None, noise_dimension], name='noise')

def discriminator(x):
    with tf.name_scope('discriminator'):
        current_input = x
        with tf.variable_scope('discriminator'):
            for layer_i, n_output in enumerate(discr_dimensions[1:]):
                n_input = int(current_input.get_shape()[1])
                W = tf.get_variable('weight' + str(layer_i), shape)
                b = tf.get_variable('bias' + str(layer_i), initializer=tf.constant_initializer(0), shape=[n_output])
                current_input = tf.nn.elu(tf.add(tf.matmul(current_input, W), b),
                                          name='discrlayer' + str(layer_i))
            n_input = int(current_input.get_shape()[1])
            n_output = int(discr_dimensions[-1]))
            W = tf.Variable(xavier_init(n_input, n_output), name='weight')
            b = tf.Variable(tf.zeros(shape=(1, n_output)), name='bias')
            D = tf.add(tf.matmul(current_input, W), b)
            D_logit = tf.nn.sigmoid(D, name = 'discr_sig_layer')
            return D_logit

def generator(z):
    with tf.name_scope('generator'):
        current_input = z
        with tf.variable_scope('generator'):
            for layer_i, n_output in enumerate(gener_dimensions[1:]):
                    n_input = int(current_input.get_shape()[1])
                    W = tf.Variable(xavier_init(n_input, n_output), name='weight' + str(layer_i))
                    b = tf.Variable(tf.zeros(shape=(1, n_output)), name='bias' + str(layer_i))
                    current_input = tf.nn.elu(tf.add(tf.matmul(current_input, W), b),
                                              name='generlayer' + str(layer_i))
                n_input = int(current_input.get_shape()[1])
                n_output = int(gener_dimensions[-1])
                W = tf.Variable(xavier_init(n_input, n_output), name='weight')
                b = tf.Variable(tf.zeros(shape=(1, n_output)), name='bias')
                G = tf.add(tf.matmul(current_input, W), b)
                G_logit = tf.nn.sigmoid(G, name = 'generator_sig_layer')
            return G_logit


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


G_sample = generator(noise)
D_logit_real = discriminator(example_image)
D_logit_fake = discriminator(G_sample)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_real,
                                                                      tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_fake,
                                                                     tf.zeros_like(D_logit_fake)))
D_loss = D_loss_fake + D_loss_real
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_fake, tf.ones_like(D_logit_fake)))
D_solver = tf.train.AdamOptimizer(0.001).minimize(D_loss)
G_solver = tf.train.AdamOptimizer(0.001).minimize(G_loss)

# Import mnist data
mnist = input_data.read_data_sets('./mnist/', one_hot=True)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch_i in range(30):
    for batch_i in range(100):
        noise_samples = np.random.normal(size=(100, noise_dimension))
        batch_xs, _ = mnist.train.next_batch(100)

        _, D_loss_curr = sess.run(fetches = [D_solver, D_loss], feed_dict={example_image: batch_xs, noise: noise_samples})
        _, G_loss_curr = sess.run(fetches = [G_solver, G_loss], feed_dict={noise:noise_samples})

    print('D_loss :{}'.format(D_loss_curr))
    print('G_loss :{}'.format(G_loss_curr))

    samples = sess.run(G_sample, feed_dict={noise: np.random.normal(size=(10, noise_dimension))})
    fig = plot(samples)
    plt.savefig('out/{}.png'.format(str(epoch_i).zfill(3)), bbox_inches='tight')
    plt.close(fig)
    print()
