from nnet_base import *
import time
import numpy as np
import matplotlib as plt

class VAE(NNbase):
    """
        NN class for VAE
    """
    def activation(self, input, name):
        return tf.nn.elu(input, name=name)

    def preproc(self, input, mean_values=[0]):
        ## Preprocessing
        with tf.variable_scope('preproc'):
            return tf.nn.bias_add(self.x, [-1 * x for x in mean_values], name='mean_sub')

    def encoder_net(self, input_shape, n_z, layers, mean_values=[0], batch_norm=False, debug=False):
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
        # Mean parameters
        self.z_mean = self.fc_lyr(prev_lyr,
                                out_num=n_z,
                                name=("z_mean" % lyr_i),
                                keep_prob=self.keep_prob,
                                batch_norm=batch_norm,
                                activation_on=False,
                                phase_train=self.phase_train)

        # Mean parameters
        self.z_sigma = self.fc_lyr(prev_lyr,
                                out_num=n_z,
                                name=("z_sigma" % lyr_i),
                                keep_prob=self.keep_prob,
                                batch_norm=batch_norm,
                                activation_on=False,
                                phase_train=self.phase_train)


    def decoder_net(self, input_shape, n_z, layers, mean_values=[0], batch_norm=False, debug=False):
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
        # Mean parameters
        self.z_mean = self.fc_lyr(prev_lyr,
                                out_num=n_z,
                                name=("fc%02d" % lyr_i),
                                keep_prob=self.keep_prob,
                                batch_norm=batch_norm,
                                activation_on=False,
                                phase_train=self.phase_train)

        # Mean parameters
        self.z_sigma = self.fc_lyr(prev_lyr,
                                out_num=n_z,
                                name=("fc%02d" % lyr_i),
                                keep_prob=self.keep_prob,
                                batch_norm=batch_norm,
                                activation_on=False,
                                phase_train=self.phase_train)


    ## Preinitialization (defines inputs)
    def create_inputs(self, input_shape, out_num, debug=False):
        with tf.variable_scope('in'):
            # placeholders for input and output
            self.x = tf.placeholder(tf.float32, shape=input_shape, name='feat')
            if debug:
                self.x = tf.Print(self.x, [tf.shape(self.x)],
                                  message='%s shape: ' % 'Input',
                                  summarize=4, first_n=1)

            self.label_hard = tf.placeholder(tf.int32, shape=[input_shape[0]], name='label_hard')

            self.phase_train = tf.placeholder(tf.bool, name='phase_train')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        tf.add_to_collection('in', self.x)
        tf.add_to_collection('in', self.label_hard)
        tf.add_to_collection('in', self.phase_train)
        tf.add_to_collection('in', self.keep_prob)

        print 'Shape of ', 'input', ' ', self.x.get_shape()
        print 'Shape of ', 'labels', ' ', self.label_hard.get_shape().as_list()

    def init_reguralizer(self, weight_decay, weight_norm='l2'):
        with tf.variable_scope('loss'):
            self.weights_norm = tf.reduce_sum(
                input_tensor=weight_decay * tf.pack(
                    [tf.nn.l2_loss(i) for i in tf.get_collection('weights')]
                ),
                name='weights_norm'
            )
        tf.add_to_collection('w_losses', self.weights_norm)

    def init_loss(self):
        with tf.variable_scope('loss'):
            # Hard label loss
            self.loss_hard = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.y, self.label_hard),
                                            name='loss_hard')
            self.loss_hard_w = self.loss_hard

        # tf.scalar_summary("loss/loss_soft", self.loss_soft)
        tf.scalar_summary("loss/loss_hard", self.loss_hard)
        tf.add_to_collection('losses', self.loss_hard)
        tf.add_to_collection('w_losses', self.loss_hard_w)

    def init_metrics(self):
        with tf.variable_scope('out'):
            # Metric
            self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y, 1), tf.int32), self.label_hard,
                                               name='correct_pred')
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accuracy')

        tf.scalar_summary("accuracy", self.accuracy)
        tf.add_to_collection('out', self.accuracy)

    ## Training initialization (defines losses and optimizer)
    def init_training(self, weight_decay, weight_norm='l2'):
        self.init_metrics()
        self.init_reguralizer(weight_decay, weight_norm)
        self.init_loss()

        with tf.variable_scope('out'):
            # Final loss is a sum of weighted losses
            self.loss = tf.add_n(tf.get_collection('w_losses'), name='loss')

        with tf.variable_scope('train'):
            # Hard labels only
            self.train_step = tf.train.RMSPropOptimizer(1e-3, momentum=0).minimize(self.loss)
            # self.train_step = tf.train.AdagradOptimizer(1e-3).minimize(self.loss)
            # self.train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(self.loss)

        tf.scalar_summary("loss/loss", self.loss)
        tf.add_to_collection('out', self.loss)

        tf.add_to_collection('train', self.train_step)

    ###################################################################################################################

    ## Training validation (generates summary)
    def train_step_run(self, data, label, params):
        # print 'NNET: Train_step: Feat type = ', data.dtype, ' feat_shape=', data.shape
        start_time_inference = time.time()
        self.sess.run(self.train_step, \
                      feed_dict={ \
                          self.x: data, \
                          self.label_hard: label[0], \
                          self.keep_prob: params['nnet_keep_prob'], \
                          self.phase_train: True})
        # self.train_step.run(feed_dict={
        #     self.x: data,
        #     self.label_hard: label[0],
        #     self.keep_prob: params['nnet_keep_prob'],
        #     self.phase_train: True})
        inf_time = time.time() - start_time_inference
        return inf_time

    ## Validation function (no summary needed)
    def train_run(self, data, label, params):
        start_time_inference = time.time()
        pred, loss, summary_cur = self.sess.run( \
            [self.pred, self.loss, self.summary_all], \
            feed_dict={self.x: data, \
                       self.label_hard: label[0], \
                       self.keep_prob: 1.0, \
                       self.phase_train: False} \
            )
        inf_time = time.time() - start_time_inference
        error, pred_val = self.get_error(pred, label[1], params['dim_angular'])

        # print 'TRAIN: pred = ', np.argmax(pred, axis=1)
        # print 'TRAIN: label = ', label[0]

        return error, pred_val, pred, loss, inf_time, summary_cur

    ## Validation function
    def val_run(self, data, label, params):
        start_time_inference = time.time()
        pred, loss, accuracy = self.sess.run( \
            [self.pred, self.loss, self.accuracy], \
            feed_dict={self.x: data, \
                       self.label_hard: label[0], \
                       self.keep_prob: 1.0, \
                       self.phase_train: False} \
            )
        inf_time = time.time() - start_time_inference
        error, pred_val = self.get_error(pred, label[1], params['dim_angular'])

        # print 'pred = ', pred
        # print 'label = ', label[0]
        # print 'gt = ', label[1]
        # print 'VAL: pred = ', np.argmax(pred, axis=1)
        # for i in range(0, y.shape[0]):
        #     print y[i], ' ',
        # print
        # print 'VAL: label = ', label[0]

        return error, pred_val, pred, loss, inf_time, accuracy

    ## Just prediction function
    def pred_run(self, data, params):
        start_time_inference = time.time()
        pred = self.sess.run( \
            [self.pred], \
            feed_dict={self.x: data, \
                       self.keep_prob: 1.0, \
                       self.phase_train: False}
        )
        inf_time = time.time() - start_time_inference
        return pred, inf_time

    def setBinValues(self, bin_values):
        self.bin_values = bin_values

    ## Pred to values conversion
    def pred2val(self, pred):
        # print 'Pred = ', pred, ' shape =', pred.shape, ' type=', type(pred)
        pred_label = np.argmax(pred, axis=1)
        return self.bin_values[pred_label]

    def pred2lbl(self, pred):
        return np.argmax(pred, axis=1)

    ## Test calculation
    def get_error(self, pred, gt, angular):
        pred_val = self.pred2val(pred)
        error = gt - pred_val
        # if we have angular coordinates we do correction, because abs error cannot be > 180 deg
        if angular:
            err_wrong_mask = np.abs(error) > 180.0
            err_wrong = error[err_wrong_mask]
            error[err_wrong_mask] = -np.sign(err_wrong) * (360.0 - np.abs(err_wrong))
        return error, pred_val

    ## Testing
    def test_run(self, data, label, params):
        pred, inf_time = self.pred_run(data, params)
        error, pred_val = self.get_error(pred, label[1], params['dim_angular'])
        return error, pred_val, pred, inf_time

    def init_session(self):
        self.sess = tf.InteractiveSession()

    ## Class constructor
    # @param input_shape  [h,w] of the input features
    # @param out_num  Number of output classes
    # @param layers   list containing number of neurons per layer
    def __init__(self, input_shape=None, out_num=None, layers=None, n_z=20, batch_norm=False, debug=False, weight_decay=0.0005,
                 weight_norm='l2', init=True):
        # Calling constructor of the base class
        NNbase.__init__(self)

        # Extracting the class name
        self.name = self.__class__.__name__

        # Init
        if init:

            # Creating inputs and groundtruth placeholders
            self.create_inputs(input_shape, out_num, debug=debug)

            # Preprocessing
            self.x_postproc = self.preproc(self.x, mean_values=[0])

            self.build_model(input_shape=[None, input_shape[0], input_shape[1], 1],
                             out_num=out_num,
                             layers=layers,
                             batch_norm=batch_norm,
                             debug=debug)

            self.init_training(weight_decay=weight_decay, weight_norm=weight_norm)

            # Some statistics:
            print 'NNET: Parameters num = ', self.params_num

            # start session
            print 'NNET: Initializing default session ...'
            self.sess = tf.InteractiveSession()

            # All summaries
            self.summary_all = tf.merge_all_summaries()