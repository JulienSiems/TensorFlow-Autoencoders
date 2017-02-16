import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from mlp2 import MLP

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

n_samples = int(mnist.train.num_examples)
training_epochs = 50
batch_size = 128
display_step = 1
logs_path = '/tmp/tensorflow_logs/example'

mlp = MLP()

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size=batch_size)
        cost = mlp.partial_fit(batch_xs, batch_ys)
        avg_cost += cost/n_samples*batch_size

        if i % 1000 == 0:
            print(mlp.accuracy_f(batch_xs, batch_ys))

    # Display logs per epoch step
    if epoch % display_step == 0:
        print(avg_cost)


