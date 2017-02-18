import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt


def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)

def autoencoder(dimensions = [784, 500, 200, 64]):
    x = tf.placeholder(dtype = tf.float32, shape=[None, dimensions[0]], name = 'x')

    current_input = x

    # ENCODER
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(xavier_init(n_input, n_output))
        b = tf.Variable(tf.truncated_normal(shape=[n_output], mean = 0.1, dtype = tf.float32))
        encoder.append(W)
        current_input = tf.nn.elu(tf.add(tf.matmul(current_input, W), b), name = 'enclayer' + str(layer_i))

    # latent representation
    z = current_input

    encoder.reverse()

    # DECODER
    count = 0
    for W in encoder:
        W_t = tf.transpose(W)
        b = tf.Variable(tf.truncated_normal(shape = [int(W_t.get_shape()[1])], mean = 0.1, dtype = tf.float32))
        current_input = tf.nn.elu(tf.add(tf.matmul(current_input, W_t), b), name = 'declayer' + str(count))
        count += 1

    y = current_input

    cost = tf.pow(tf.sub(y, x), 2)
    tf.summary.scalar('cost', cost)

    return {'x':x, 'y':y, 'cost':cost}


def testmnist():
    ae = autoencoder()
    mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
    mean_img = np.mean(mnist.train.images, axis=0)

    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(ae['cost'])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 50
    n_epochs = 10
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img-mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict = {ae['x']:train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    # %%
    # Plot example reconstructions
    n_examples = 15
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape([recon[example_i, :] + mean_img], (28, 28)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()

    # Save the completed Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, "/tmp/autoencoder.ckpt")
    print("Model saved in file: %s" % save_path)

if __name__=='__main__':
    testmnist()