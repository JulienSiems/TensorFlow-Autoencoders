import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import argparse
import sys


from tensorflow.examples.tutorials.mnist import input_data

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    ## INIT
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.placeholder(tf.float32, [784, 10])
    b = tf.placeholder(tf.float32, [10])

    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for _ in range(1000):
        # This piece of code retrieves images and the corresponding labels
        # from the the mnist data base
        batch_xs, batch_ys = mnist.train.next_batch(100)

        # Runs one step of the Gradientdescentoptimizer
        sess.run(fetches=train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
