import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)
def autoencoder():
    image = tf.placeholder(tf.float32, [None, 784])

    w1 = tf.Variable(xavier_init(784, 500))
    b1 = tf.Variable(tf.truncated_normal(shape = [500], mean=0.1, dtype=tf.float32))

    w2 = tf.Variable(xavier_init(500, 64))
    b2 = tf.Variable(tf.truncated_normal(shape = [64], mean=0.1, dtype=tf.float32))

    w3 = tf.Variable(xavier_init(64, 500))
    b3 = tf.Variable(tf.truncated_normal(shape = [500], mean=0.1, dtype=tf.float32))

    w4 = tf.Variable(xavier_init(500, 784))
    b4 = tf.Variable(tf.truncated_normal(shape = [784], mean=0.1, dtype=tf.float32))

    layer1 = tf.nn.softplus(tf.add(tf.matmul(image, w1), b1), name = 'layer1')
    layer2 = tf.nn.softplus(tf.add(tf.matmul(layer1, w2), b2), name = 'layer2')
    layer3 = tf.nn.softplus(tf.add(tf.matmul(layer2, w3), b3), name = 'layer3')
    reconstruction = tf.nn.softplus(tf.add(tf.matmul(layer3, w4), b4), name = 'layer4')

    cost = tf.reduce_sum(tf.pow(tf.sub(reconstruction, image), 2))
    return {'image':image, 'reconstruction':reconstruction, 'cost':cost}

def test_mnist():
    ae = autoencoder()

    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(ae['cost'])

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(10):
      images = mnist.test.images
      error = sess.run(ae['cost'], {ae['image']: images})
      print('Test error {:6.2f}%'.format(error))
      for batch in range(60):
        images, _ = mnist.train.next_batch(100)
        sess.run(optimizer, {ae['image']: images})

    # Plot example reconstructions
    n_examples = 2
    test_xs1, _ = mnist.test.next_batch(1)
    recon1 = sess.run(ae['reconstruction'], feed_dict={ae['image']: test_xs1})
    test_xs2, _ = mnist.test.next_batch(1)
    recon2 = sess.run(ae['reconstruction'], feed_dict={ae['image']: test_xs2})
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    axs[0][0].imshow(
        np.reshape(test_xs1[:, :], (28, 28)))
    axs[1][0].imshow(
        np.reshape([recon1[:, :]], (28, 28)))
    axs[0][1].imshow(
        np.reshape(test_xs2[:, :], (28, 28)))
    axs[1][1].imshow(
        np.reshape([recon2[:, :]], (28, 28)))
    fig.show()
    #plt.savefig(fig)
    plt.draw()
    plt.waitforbuttonpress()

if __name__=='__main__':
    test_mnist()