from vae import *

# Loading MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

# Initialization of random number generator
np.random.seed(0)
tf.set_random_seed(0)

