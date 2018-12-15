import tensorflow as tf
from numpy import prod
import numpy as np
from math import ceil
import scipy
import scipy.stats as st
import math


class NNbase(object):
    """
    Base class for all neural nets. 
    Contains useful methods to build NN models.
    """

    ## Constructor
    def __init__(self):
        self.params_num = 0

    @staticmethod
    def xavier_init(shape, n_inputs, n_outputs, uniform=True):
        """
        Set the parameter initialization using the method described.
        This method is designed to keep the scale of the gradients roughly the same
        in all layers.
        Xavier Glorot and Yoshua Bengio (2010):
                 Understanding the difficulty of training deep feedforward neural
                 networks. International conference on artificial intelligence and
                 statistics.
        Args:
          shape: Parameter shape
          n_inputs: The number of input nodes into each output.
          n_outputs: The number of output nodes for each input.
          uniform: If true use a uniform distribution, otherwise use a normal.
        Returns:
          An initializer.
        """
        if uniform:
            # 6 was used in the paper.
            init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
            return tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
        else:
            # 3 gives us approximately the same limits as above since this repicks
            # values greater than 2 standard deviations from the mean.
            stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
            return tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)


    ## Weight initialization
    def weight_variable(self, shape, name='W', stddev=0.1):
        """
        A wrapper for all weight variables
        """
        initial = tf.truncated_normal(shape, stddev=stddev)
        weights = tf.Variable(initial, name=name)
        tf.add_to_collection('weights', weights)
        hist_summary = tf.summary.histogram("hist/" + tf.get_variable_scope().name + "/" + name, weights)
        self.params_num += np.prod(shape)
        return weights

    def weight_variable_xavier(self, shape, name='W', uniform=False):
        """
        Generates xavier initialized weights for CNN and FC layers
        """
        # The type of the layer defined based on shape
        if len(shape) == 4:
            num_in  = shape[2]
            num_out = shape[3]
        elif len(shape) == 2:
            num_in  = shape[0]
            num_out = shape[1]
        else:
            raise ValueError('XAVIER_W: shape is neither 2 nor 4 dimensional !')

        initial = NNbase.xavier_init(shape, n_inputs=num_in, n_outputs=num_out, uniform=uniform)
        weights = tf.get_variable(name, initializer=initial)
        tf.add_to_collection('weights', weights)
        hist_summary = tf.summary.histogram("hist/" + tf.get_variable_scope().name + "/" + name, weights)
        self.params_num += np.prod(shape)
        return weights

    def bias_variable(self, shape, name="B", init_bias=0.1):
        initial = tf.constant(init_bias, shape=shape)
        biases = tf.get_variable(name, initializer=initial)
        tf.add_to_collection('weights', biases)
        hist_summary = tf.summary.histogram("hist/" + tf.get_variable_scope().name + "/" + name, biases)
        self.params_num += np.prod(shape)
        return biases

    @staticmethod
    def batch_norm(x, phase_train, name="bn", decay=0.9):
        """
        Batch normalization on convolutional maps.
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            name:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope(name):
            # Depending on what input shape we have we keep different axises
            # when calculating moments
            x_shape = x.get_shape().as_list()
            if len(x_shape) == 4:
                keep_axis = [0,1,2]
                n_out = x_shape[3]
            elif len(x_shape) == 2:
                keep_axis = [0]
                n_out = x_shape[1]
            else:
                raise ValueError('BATCHNORM: shape of tensor for tf.nn.moments is neither 2 nor 4 dimensional !')

            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma', trainable=True)

            batch_mean, batch_var = tf.nn.moments(x, keep_axis, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=decay)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='bn_op')
        return normed


    def get_bilinear_filter(self, f_shape, name="W"):
        """
        Initialization of a deconvolution filter to bilinear filter
        """
        width = f_shape[0]
        heigh = f_shape[1]
        f = ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        # print f_shape
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant(value=weights, dtype=tf.float32)
        weights = tf.Variable(init, dtype=tf.float32, name=name)
        hist_summary = tf.summary.histogram("hist/" + tf.get_variable_scope().name + "/" + name, weights)

        return  weights

    @staticmethod
    def gauss_kern(kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel.astype(np.float32)
        return kernel

    def activation(self, input):
        """
        Default activation wrapper
        """
        return tf.nn.relu(input, name='activation')

    def fc_lyr(self, input, out_num, name, keep_prob=0.5, activation_on=True, batch_norm=False, phase_train=None, debug=True):
        """
        Fully connected layer
        @param: input: (2D Tensor) input tensor
        @param: out_num: (int) size of the output
        @param: name: (str) name of the FC layer
        @param: keep_prob: (float) keep probability for the dropout
        @param: activation_on: (bool) add the action or not
        @param: batch_norm: (bool) adding batch normalization or not
        @param: phase_train: (bool Tensor) trainin/testing phase indicator
        @param: debug: (bool) show debug information
        """
        with tf.variable_scope(name):

            # Flattening first (if required)
            input_shape = input.get_shape().as_list()
            if len(input_shape) > 2:
                if len(input_shape) == 4:
                    input = tf.reshape(input, [-1, input_shape[1] * input_shape[2] * input_shape[3]])
                elif len(input_shape) == 3:
                    input = tf.reshape(input, [-1, input_shape[1] * input_shape[2]])

            # Batch normalization (if required)
            if batch_norm:
                input = self.batch_norm(input, phase_train=phase_train)

            prev_lyr_shape = input.get_shape().as_list()
            in_num = prev_lyr_shape[1]

            b_fc = self.bias_variable([out_num])
            W_fc = self.weight_variable_xavier([in_num, out_num])
            if activation_on:
                h_fc = self.activation(tf.matmul(input, W_fc, name='matmul') + b_fc, name='activation')
            else:
                h_fc = tf.matmul(input, W_fc, name='matmul') + b_fc

            h_fc_drop = tf.nn.dropout(h_fc, keep_prob=keep_prob, name='dropout')
            if debug:
                h_fc_drop = tf.Print(h_fc_drop, [tf.shape(h_fc_drop)],
                                  message='%s: shape = ' % name,
                                  summarize=4, first_n=1)
            print(('%s: in_num = %d, out_num = %d ' % (name, in_num, out_num)))

        return h_fc_drop