#!/usr/bin/env python
import argparse
from matplotlib import pyplot as plt
from vae import *


# Loading MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

# Initialization of random number generator
np.random.seed(0)
tf.set_random_seed(0)


def train(network_architecture, learning_rate=1e-3,
          batch_size=100, training_epochs=10, display_step=5):

    vae = VAE(input_shape=(None, 784),
             architecture=network_architecture,
             batch_size=batch_size,
             learning_rate=learning_rate,
             batch_norm=False,
             debug=True)

    params = {}
    params['keep_prob'] = 1.0

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost, time = vae.train_step_run(batch_xs, params=params)

            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost)
    return vae

def test_reconstruct(vae):
    x_sample = mnist.test.next_batch(100)[0]
    x_rec, x_rec_inftime = vae.reconstruct(x_sample)
    print('Test rec: sample shape = ', x_sample.shape, ' x_rec_shape = ', x_rec.shape)

    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(x_sample[i][:].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(x_rec[i][:].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    plt.show(block=True)


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Parameters
    architecture = {}
    architecture['encoder'] = [500, 500, 250]
    architecture['decoder'] = [250, 500, 500]
    architecture['n_z'] = 20

    # Start training
    vae = train(network_architecture=architecture,
                training_epochs=15,
                display_step=1)

    # Testing
    test_reconstruct(vae)


if __name__ == "__main__":
    main()