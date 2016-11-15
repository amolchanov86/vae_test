

from nnet_base import *
import time
import numpy as np

class VAE(NNbase):
    """
        NN class for VAE
    """
    ## Activation function wrapper
    def activation(self, input, name):
        return tf.nn.elu(input, name=name)
        # return tf.nn.softplus(input, name=name)

    ## Preprocessing of features
    def preproc(self, input, mean_values=[0]):
        ## Flatenning
        input_shape = input.get_shape().as_list()
        if len(input_shape) > 2:
            if len(input_shape) == 4:
                input = tf.reshape(input, [-1, input_shape[1] * input_shape[2] * input_shape[3]])
            elif len(input_shape) == 3:
                input = tf.reshape(input, [-1, input_shape[1] * input_shape[2]])

        ## Preprocessing
        with tf.variable_scope('preproc'):
            input_shape = input.get_shape().as_list()
            if len(mean_values) == 1:
                mean_values = mean_values * np.ones([input_shape[-1]])
            # return tf.nn.bias_add(input, [-1 * x for x in mean_values], name='mean_sub')
            return tf.nn.bias_add(input, -mean_values, name='mean_sub')

    ## Encoder net
    def encoder_net(self, n_z, layers, batch_norm=False, debug=True):
        # Current previous layer
        prev_lyr = self.x_postproc

        # Building layers
        # Every previous layer should be flattened before entering here
        for lyr_i in range(0, len(layers)):
            h_fc_drop = self.fc_lyr(prev_lyr,
                                    out_num=layers[lyr_i],
                                    name=("fc%02d" % lyr_i),
                                    keep_prob=self.keep_prob,
                                    batch_norm=batch_norm,
                                    phase_train=self.phase_train)

            prev_lyr = h_fc_drop

        # --- Creating latent space
        # Mean of latent variables
        self.z_mean = self.fc_lyr(prev_lyr,
                                out_num=n_z,
                                name="z_mean",
                                keep_prob=self.keep_prob,
                                batch_norm=batch_norm,
                                activation_on=False,
                                phase_train=self.phase_train)

        # log(sigma^2) of latent variables (here I use relu to get rid of negative part, since it can not be negative)
        self.z_log_sigma2 = tf.nn.relu(self.fc_lyr(prev_lyr,
                                        out_num=n_z,
                                        name="z_log_sigma2_logit",
                                        keep_prob=self.keep_prob,
                                        batch_norm=batch_norm,
                                        activation_on=False,
                                        phase_train=self.phase_train), name='z_log_sigma2')

        # Printing shapes
        if debug:
            print ('Debug printing ...')
            print ('z_mean shape = ', self.z_mean.get_shape().as_list())
            print ('z_log_sigma2 shape = ', self.z_log_sigma2.get_shape().as_list())

            self.z_mean = tf.Print(self.z_mean, [tf.shape(self.z_mean)],
                                 message='%s: shape = ' % 'z_mean',
                                 summarize=4, first_n=1)

            self.z_log_sigma2 = tf.Print(self.z_log_sigma2, [tf.shape(self.z_log_sigma2)],
                                 message='%s: shape = ' % 'z_log_sigma2',
                                 summarize=4, first_n=1)

        return (self.z_mean, self.z_log_sigma2)


    ## Decoder net
    def decoder_net(self, layers, batch_norm=False, debug=True):
        # Necessary parameters
        x_post_shape = self.x_postproc.get_shape().as_list()

        # Take converted sample
        prev_lyr = self.z

        # Building layers
        # Every previous layer should be flattened before entering here
        for lyr_i in range(0, len(layers)):
            h_fc_drop = self.fc_lyr(prev_lyr,
                                    out_num=layers[lyr_i],
                                    name=("fc%02d" % lyr_i),
                                    keep_prob=self.keep_prob,
                                    batch_norm=batch_norm,
                                    phase_train=self.phase_train)

            prev_lyr = h_fc_drop

        # --- Creating output space
        # Reconstructed input
        self.x_rec = tf.nn.sigmoid(self.fc_lyr(prev_lyr,
                                out_num=x_post_shape[1],
                                name="x_rec_logit",
                                keep_prob=self.keep_prob,
                                batch_norm=batch_norm,
                                activation_on=False,
                                phase_train=self.phase_train), name='x_rec')
        return self.x_rec


    ## Preinitialization (defines inputs)
    def create_inputs(self, input_shape, debug=False):
        with tf.variable_scope('in'):
            # placeholders for input and output
            self.x = tf.placeholder(tf.float32, shape=input_shape, name='feat')
            if debug:
                self.x = tf.Print(self.x, [tf.shape(self.x)],
                                  message='%s shape: ' % 'Input',
                                  summarize=4, first_n=1)

            self.phase_train = tf.placeholder(tf.bool, name='phase_train')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        tf.add_to_collection('in', self.x)
        tf.add_to_collection('in', self.phase_train)
        tf.add_to_collection('in', self.keep_prob)

        print 'Shape of ', 'input', ' ', self.x.get_shape().as_list()

    ## Initializes reguralizer (if needed)
    def init_reguralizer(self, weight_decay, weight_norm='l2'):
        with tf.variable_scope('loss'):
            self.weights_norm = tf.reduce_sum(
                input_tensor=weight_decay * tf.pack(
                    [tf.nn.l2_loss(i) for i in tf.get_collection('weights')]
                ),
                name='weights_norm'
            )
        tf.add_to_collection('w_losses', self.weights_norm)
        print ('Reguralizer is not set up')

    ## Initialization of losses
    def init_loss(self, n_z):
        with tf.variable_scope('loss'):
            # Encoder loss (latent loss)
            self.loss_enc = 0.5 * tf.reduce_sum(tf.reduce_sum(tf.exp(self.z_log_sigma2), 1)
                                        + tf.reduce_sum(tf.square(self.z_mean), 1)
                                        - n_z
                                        - tf.reduce_sum(self.z_log_sigma2, 1),
                                                0, name='loss_enc')

            # self.loss_dec = 0.5 * tf.reduce_mean(tf.square(tf.sub(self.x_postproc, self.x_rec, name='error')), name='loss_rec') / self.x_rec_sigma2

            self.loss_dec = -tf.reduce_sum(self.x_postproc * tf.log(1e-10 + self.x_rec)
                            + (1 - self.x_postproc) * tf.log(1e-10 + 1 - self.x_rec))

        tf.scalar_summary("loss/loss_enc", self.loss_enc)
        tf.scalar_summary("loss/loss_dec", self.loss_dec)

        tf.add_to_collection('losses', self.loss_enc)
        tf.add_to_collection('losses', self.loss_dec)

        # Weighted losses will be just summed up together
        # to create the final loss for optimization
        tf.add_to_collection('w_losses', self.loss_enc)
        tf.add_to_collection('w_losses', self.loss_dec)

    ## Initialization of metrics
    def init_metrics(self):
        print ('Metrics are not set up')

    ## Training initialization (defines losses and optimizer)
    def init_training(self, weight_decay=0.0005, weight_norm='l2'):
        self.init_metrics()
        # self.init_reguralizer(weight_decay, weight_norm)
        self.init_loss(self.architecture['n_z'])

        with tf.variable_scope('out'):
            # Final loss is a sum of weighted losses
            self.loss = tf.add_n(tf.get_collection('w_losses'), name='loss')

        with tf.variable_scope('train'):
            # Hard labels only
            self.train_step = tf.train.RMSPropOptimizer(self.learning_rate, momentum=0).minimize(self.loss)
            # self.train_step = tf.train.AdagradOptimizer(1e-3).minimize(self.loss)
            # self.train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(self.loss)

        tf.scalar_summary("loss/loss", self.loss)
        tf.add_to_collection('out', self.loss)

        tf.add_to_collection('train', self.train_step)

    ###################################################################################################################
    def train_step_run(self, X, params):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        start_time_inference = time.time()
        opt, loss = self.sess.run((self.train_step, self.loss),
                                  feed_dict={self.x: X,
                                             self.keep_prob: params['keep_prob'],
                                             self.phase_train: True})
        inf_time = time.time() - start_time_inference
        return loss, inf_time

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        start_time_inference = time.time()
        z_mean = self.sess.run(self.z_mean,
                                feed_dict = {self.x: X,
                                self.keep_prob: 1.0,
                                self.phase_train: False})
        inf_time = time.time() - start_time_inference
        return z_mean, inf_time

    def generate(self, z_mu=None ):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        start_time_inference = time.time()
        if z_mu is None:
            z_mu = np.random.normal(size=self.architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        x_rec = self.sess.run(self.x_rec,
                                feed_dict={self.z: z_mu,
                                self.keep_prob: 1.0,
                                self.phase_train: False})
        inf_time = time.time() - start_time_inference
        return x_rec, inf_time

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        start_time_inference = time.time()
        x_rec = self.sess.run(self.x_rec,
                                feed_dict = {self.x: X,
                                self.keep_prob: 1.0,
                                self.phase_train: False})
        inf_time = time.time() - start_time_inference

        return x_rec, inf_time

    def init_session(self):
        self.sess = tf.InteractiveSession()

    ## Class constructor
    # @param input_shape  [h,w] of the input features
    # @param architecture   dictionary ('encoder','decoder', 'n_z') containing lists of number of neurons in every layer
    def __init__(self, input_shape=None, architecture=None, batch_size=100, learning_rate=1e-3,
                 batch_norm=False, debug=True,
                 weight_decay=0.0005, weight_norm='l2',
                 init=True):

        # Calling constructor of the base class
        NNbase.__init__(self)

        # Extracting the class name
        self.name = self.__class__.__name__

        # Init
        if init:
            # Parameters
            self.x_rec_sigma2 = 1 # Reconstruction sigma
            self.learning_rate = learning_rate # Initial learning rate
            self.architecture = architecture

            # --- Architecture initialization
            # Creating inputs and groundtruth placeholders
            self.create_inputs(input_shape, debug=debug)
            # self.batch_size = self.x.get_shape().as_list()[0]
            self.batch_size = batch_size

            # Preprocessing
            self.x_postproc = self.preproc(self.x, mean_values=[0])

            # Initilization of net architecture
            self.encoder_net(n_z=architecture['n_z'],
                             layers=architecture['encoder'],
                             batch_norm=batch_norm, debug=debug)

            # Drawing samples for decoder (just 1 sample for now)
            self.eps = tf.random_normal(shape=(self.batch_size, architecture['n_z']),
                                   mean=0, stddev=1,
                                   dtype=tf.float32, name='eps')

            # Transforming eps to z (according to re-parametrization trick)
            # z = mu + sigma*epsilon
            self.z = tf.add(self.z_mean,
                            tf.mul(tf.sqrt(tf.exp(self.z_log_sigma2)), self.eps, name='sigmaXeps'), name='z')

            # Decoder (generator or reconstruction) network
            self.decoder_net(layers=architecture['decoder'],
                             batch_norm=batch_norm, debug=debug)

            # Initialization of training (losses, optimizer etc.)
            self.init_training(weight_decay=weight_decay, weight_norm=weight_norm)


            # --- Other initialization and printing
            # Reporting statistics:
            print 'NNET: Parameters num = ', self.params_num # Number of parameters in the network

            # start session
            print 'NNET: Initializing default session ...'
            self.sess = tf.InteractiveSession()

            # All summaries
            self.summary_all = tf.merge_all_summaries()

            # Initializing all tensors (variables)
            print 'NNET: Initializing all variables ...'
            self.sess.run(tf.initialize_all_variables())