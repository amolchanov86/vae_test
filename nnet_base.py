import tensorflow as tf
from numpy import prod
import numpy as np
from math import ceil
import scipy
import scipy.stats as st
import math

## TODO


class NNbase(object):
    """
    Base class for all neural nets. Contains useful methods to build your models
    """

    ## Constructor
    def __init__(self):
        self.params_num = 0

    @staticmethod
    def xavier_init(shape, n_inputs, n_outputs, uniform=True):
        """Set the parameter initialization using the method described.
        This method is designed to keep the scale of the gradients roughly the same
        in all layers.
        Xavier Glorot and Yoshua Bengio (2010):
                 Understanding the difficulty of training deep feedforward neural
                 networks. International conference on artificial intelligence and
                 statistics.
        Args:
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
        initial = tf.truncated_normal(shape, stddev=stddev)
        # tf.get_variable_scope() #returns the curent scope
        weights = tf.Variable(initial, name=name)
        tf.add_to_collection('weights', weights)
        hist_summary = tf.histogram_summary("hist/" + tf.get_variable_scope().name + "/" + name, weights)

        self.params_num += np.prod(shape)
        return weights

    def weight_variable_xavier(self, shape, name='W', uniform=False):
        ## Generates xavier initialized weights for cnn and fc
        # type of layer defined based on shape
        if len(shape) == 4:
            num_in  = shape[2]
            num_out = shape[3]
        elif len(shape) == 2:
            num_in  = shape[0]
            num_out = shape[1]
        else:
            raise ValueError('XAVIER_W: shape is neither 2 nor 4 dimensional !')

        initial = NNbase.xavier_init(shape, n_inputs=num_in, n_outputs=num_out, uniform=uniform)
        weights = tf.Variable(initial, name=name)
        tf.add_to_collection('weights', weights)
        hist_summary = tf.histogram_summary("hist/" + tf.get_variable_scope().name + "/" + name, weights)

        self.params_num += np.prod(shape)

        return weights

    def bias_variable(self, shape, name="B", init_bias=0.1):
        initial = tf.constant(init_bias, shape=shape)
        biases = tf.Variable(initial, name=name)
        tf.add_to_collection('weights', biases)
        hist_summary = tf.histogram_summary("hist/" + tf.get_variable_scope().name + "/" + name, biases)

        self.params_num += np.prod(shape)

        return biases

    ## Batch normalizaton for convolutions
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
        # print 'Enteringn scope ', name
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


    ## Initialization of a deconvolution filter to bilinear filter
    def get_bilinear_filter(self, f_shape, name="W"):
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
        hist_summary = tf.histogram_summary("hist/" + tf.get_variable_scope().name + "/" + name, weights)

        return  weights

    # Gaussian kernel
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

    @staticmethod
    def conv2d_sparsdense(input, filter, filter_shape, strides, padding, name=None, debug=False):
        # input = tf.Print(input, [tf.shape(input)],
        #                  message='Shape of %s in feature map' % name,
        #                  summarize=4, first_n=1)

        # print 'Shape of ', name, ' in feat map = ', input.get_shape()

        if name == None:
            name = ''

        with tf.variable_scope(name):
            patch_size = [1, filter_shape[0], filter_shape[1], 1]

            # Exctracted shape
            # N x H x W x (length of the patch produced, i.e. flattened patch length)
            patches = tf.extract_image_patches(input, padding=padding, ksizes=patch_size, strides=strides,
                                               rates=[1, 1, 1, 1], name='patches')
            patches_shape = tf.shape(patches)
            # patches_shape = tf.Print(patches_shape, [tf.shape(patches)],
            #                          message='Shape of %s out feature map' % 'patches_shape',
            #                          summarize=4, first_n=1)

            # Dimensions of patches: N * H *W * Patch_length,
            # where N * H * W = number of pathces produced,
            # Pathces_length = number of elelments in the patch

            patches_mx = tf.reshape(patches,
                                    shape=[patches_shape[0] * patches_shape[1] * patches_shape[2], patches_shape[3]],
                                    name='patches_mx')

            # patches_mx = tf.transpose(patches_mx)

            # patches_mx = tf.Print(patches_mx, [tf.shape(patches_mx)],
            #                       message='Shape of %s out feature map' % 'patches_mx',
            #                       summarize=4, first_n=1)

            # patches_mx = tf.Print(patches_mx, [tf.shape(filter)],
            #                       message='Shape of filters %s ' % 'filter_shape',
            #                       summarize=4, first_n=1)

            # Sparse-dense multiplication
            feat_map_flat = tf.matmul(patches_mx, filter, b_is_sparse=True)
            # feat_map = tf.matmul(patches, filter, b_is_sparse=True)
            # feat_map_flat = tf.truncated_normal([patches_shape[0] * patches_shape[1] * patches_shape[2], tf.shape(filter)[1]], stddev=0.1)
            # feat_map_flat = tf.zeros([patches_shape[0] * patches_shape[1] * patches_shape[2], tf.shape(filter)[1]])

            # Trasposing and reshaping back to the feature map
            # feat_map_flat = tf.transpose(feat_map_flat)
            # feat_map_flat = tf.Print(feat_map_flat, [tf.shape(feat_map_flat)],
            #                          message='Shape of %s out feature map' % 'feat_map_flat',
            #                          summarize=4, first_n=1)

            feat_map = tf.reshape(feat_map_flat,
                                  [patches_shape[0], patches_shape[1], patches_shape[2], filter_shape[3]])

            # feat_map = tf.Print(feat_map, [tf.shape(feat_map)],
            #                     message='Shape of %s out feature map' % name,
            #                     summarize=4, first_n=1)
        # print 'Shape of ', name, ' out feat map = ', feat_map.get_shape()

        return feat_map

    # Automatically initializes filters
    @staticmethod
    def conv2d_sparsedense_autofill(input, filter_shape, strides, padding, sparsity, name=None):
        filter = np.zeros(filter_shape)

        # Filling in filters
        if filter_shape[0] == 3 and filter_shape[1] == 3:
            # Create edge filter
            filter = np.zeros(filter_shape)
            filter[:, 0, :, :] = -1
            filter[:, 1, :, :] = 0
            filter[:, 2, :, :] = +1
        else:
            # Gaussian blur filter
            filter_mx_2d = NNbase.gauss_kern(max(filter_shape[0], filter_shape[1]),
                                      np.ceil(max(filter_shape[0], filter_shape[1]) / 2.0))
            # plt.imshow(filter_mx_2d)
            # plt.show()
            for ch_i in range(0, filter.shape[2]):
                for out_i in range(0, filter.shape[3]):
                    filter[:, :, ch_i, out_i] = filter_mx_2d

        filter_flat = filter.reshape([filter_shape[0] * filter_shape[1] * filter_shape[2], filter_shape[3]])

        # Collect all indices and values into lists
        filter_indx = []
        filter_val = []
        for row_i in range(0, filter_flat.shape[0]):
            for col_i in range(0, filter_flat.shape[1]):
                filter_indx.append([row_i, col_i])
                filter_val.append(filter_flat[row_i, col_i])

        # Sparsify indices and values
        numel_retain = max(np.ceil(sparsity * len(filter_val)), 1)
        all_indices = range(0, len(filter_val))
        retained_indices = np.random.choice(all_indices, size=numel_retain, replace=False)

        # Get sparcified indices,values
        filter_sp = np.zeros(filter_flat.shape, dtype=np.float32)
        for indx_i in retained_indices:
            # filter_sp_indx.append(filter_indx[indx_i])
            # filter_sp_val.append(filter_val[indx_i])
            filter_sp[filter_indx[indx_i][0], filter_indx[indx_i][1]] = filter_val[indx_i]

        # print 'Sparcified indices and values'
        # print filter_sp_indx
        # print filter_sp_val
        print 'With sparsity = ', sparsity, ' Elements retained = ', len(retained_indices), ' Elements total = ', len(
            filter_val)

        # Get sparse tensor
        w_t = tf.Variable(tf.constant(value=filter_sp, dtype=tf.float32), dtype=tf.float32, name='W')

        feat_map = NNbase.conv2d_sparsdense(input, w_t, filter_shape, strides=strides, padding=padding, name=name)
        return feat_map


    # Filter should be of the shape  [number of filters X filter length]
    # where filter length is number of elements in the filter
    @staticmethod
    def conv2d_sparse(input, filter, filter_shape, strides, padding, name='', debug=False):
        # input = tf.Print(input, [tf.shape(input)],
        #                  message='Shape of %s in feature map' % name,
        #                  summarize=4, first_n=1)

        # print 'Shape of ', name, ' in feat map = ', input.get_shape()

        with tf.variable_scope(name):
            patch_size = [1, filter_shape[0], filter_shape[1], 1]

            # Exctracted shape
            # N x H x W x (length of the patch produced, i.e. flattened patch length)
            patches = tf.extract_image_patches(input, padding=padding, ksizes=patch_size, strides=strides,
                                               rates=[1, 1, 1, 1], name='patches')
            patches_shape = tf.shape(patches)
            # patches_shape = tf.Print(patches_shape, [tf.shape(patches)],
            #                          message='Shape of %s out feature map' % 'patches_shape',
            #                          summarize=4, first_n=1)

            # Dimensions of patches: N * H *W * Patch_length,
            # where N * H * W = number of pathces produced,
            # Pathces_length = number of elelments in the patch
            patches_mx = tf.reshape(patches,
                                    shape=[patches_shape[0] * patches_shape[1] * patches_shape[2], patches_shape[3]],
                                    name='patches_mx')
            # patches_mx = tf.transpose(patches_mx)

            # patches_mx = tf.Print(patches_mx, [tf.shape(patches_mx)],
            #                       message='Shape of %s out feature map' % 'patches_mx',
            #                       summarize=4, first_n=1)

            # patches_mx = tf.Print(patches_mx, [tf.shape(filter)],
            #                       message='Shape of filters %s ' % 'filter_shape',
            #                       summarize=4, first_n=1)

            # Sparse multiplication
            feat_map_flat = tf.sparse_tensor_dense_matmul(filter, patches_mx, adjoint_b=True)

            # Trasposing and reshaping back to the feature map
            feat_map_flat = tf.transpose(feat_map_flat)
            # feat_map_flat = tf.Print(feat_map_flat, [feat_map_flat.dtype],
            #                          message='Type %s' % feat_map_flat.dtype,
            #                          summarize=4, first_n=1)

            # feat_map_flat = tf.Print(feat_map_flat, [tf.shape(feat_map_flat)],
            #                          message='Shape of %s out feature map' % 'feat_map_flat',
            #                          summarize=4, first_n=1)

            feat_map = tf.reshape(feat_map_flat,
                                  [patches_shape[0], patches_shape[1], patches_shape[2], filter_shape[3]])

            # if debug:
            #     feat_map = tf.Print(feat_map, [tf.shape(feat_map)],
            #                         message='Shape of %s out feature map' % name,
            #                         summarize=4, first_n=1)
            print 'Shape of ', name, ' = ', feat_map.get_shape()

        return feat_map

    # Automatically initializes filters
    @staticmethod
    def conv2d_sparse_autofill(input, filter_shape, strides, padding, sparsity, name=None):
        if name == None:
            name = ""
        filter = np.zeros(filter_shape, dtype=np.float32)

        # --- Filling in filters
        if filter_shape[0] == 3 and filter_shape[1] == 3:
            # Create edge filter
            filter = np.zeros(filter_shape, dtype=np.float32)
            filter[:, 0, :, :] = -1
            filter[:, 1, :, :] = 0
            filter[:, 2, :, :] = +1
        else:
            # Gaussian blur filter
            filter_mx_2d = NNbase.gauss_kern(max(filter_shape[0], filter_shape[1]),
                                      np.ceil(max(filter_shape[0], filter_shape[1]) / 2.0))
            # plt.imshow(filter_mx_2d)
            # plt.show()
            for ch_i in range(0, filter.shape[2]):
                for out_i in range(0, filter.shape[3]):
                    filter[:, :, ch_i, out_i] = filter_mx_2d

        filter_sp_shape = [filter_shape[3], filter_shape[0] * filter_shape[1] * filter_shape[2]]
        filter_flat = filter.reshape([filter_shape[0] * filter_shape[1] * filter_shape[2], filter_shape[3]])

        # --- Collect all indices and values into lists
        filter_indx = []
        filter_val = []
        for row_i in range(0, filter_flat.shape[0]):
            for col_i in range(0, filter_flat.shape[1]):
                filter_indx.append([col_i, row_i])
                filter_val.append(filter_flat[row_i, col_i])

        # Sparsify indices and values
        numel_retain = max(np.ceil(sparsity * len(filter_val)), 1)
        all_indices = range(0, len(filter_val))
        retained_indices = np.random.choice(all_indices, size=numel_retain, replace=False)

        # Get sparcified indices,values
        filter_sp_indx = []
        filter_sp_val = []
        for indx_i in range(0, len(retained_indices)):
            filter_sp_indx.append(filter_indx[indx_i])
            filter_sp_val.append(filter_val[indx_i])

        # print 'Sparcified indices and values'
        # print filter_sp_indx
        # print filter_sp_val
        print 'With sparsity = ', sparsity, ' Elements retained = ', len(filter_sp_val), ' Elements total = ', len(
            filter_val), ' type = ', type(filter_sp_val[0])

        # --- Get sparse tensor
        w_t = tf.SparseTensor(indices=filter_sp_indx, values=filter_sp_val, shape=filter_sp_shape)

        feat_map = NNbase.conv2d_sparse(input, w_t, filter_shape, strides=strides, padding=padding, name=name)
        return feat_map

    ## My convolution
    #@param filter if provided then used for initialization, otherwise filter shape is used and filters are created internally
    #@param filter_shape shape of the filter. Used when filter is not provided
    def conv2d(self, input, filter=None, filter_shape=None, strides=[1,1,1,1], padding='SAME', sparsity=None, sparsedense=True, use_cudnn_on_gpu=None, data_format=None, name=None):
        with tf.variable_scope(name):
            if sparsity == None:
                # print 'Initializing nonsparse convolutions'
                if filter != None:
                    feat_map = tf.nn.conv2d(input=input,
                                            filter=filter,
                                            strides=strides,
                                            padding=padding,
                                            use_cudnn_on_gpu=use_cudnn_on_gpu,
                                            data_format=data_format,
                                            name=name)
                else:
                    W_conv = self.weight_variable(filter_shape, name='W')
                    b_conv = self.bias_variable([filter_shape[3]], name='B')
                    feat_map = tf.nn.conv2d(input=input,
                                            filter=W_conv,
                                            strides=strides,
                                            padding=padding,
                                            use_cudnn_on_gpu=use_cudnn_on_gpu,
                                            data_format=data_format,
                                            name=name) + b_conv

            elif not sparsedense:
                print 'Initializing SPARSE convolutions'
                feat_map = NNbase.conv2d_sparse_autofill(input, filter.get_shape().as_list(), strides=strides, padding=padding, sparsity=sparsity, name=name)
            else:
                print 'Initializing SPARSE-DENSE convolutions'
                feat_map = NNbase.conv2d_sparsedense_autofill(input, filter.get_shape().as_list(), strides=strides, padding=padding, sparsity=sparsity, name=name)

        return feat_map


    ## Transpose convolutions for upsampling and deconvolution
    # @param init_type Initialization type: 'bilinear', 'gaussian' (stddev parameter is required)
    def conv2d_transpose(self, name_or_scope,
                         x, n_filters,
                         k_h=8, k_w=8,
                         stride_h=4, stride_w=4,
                         activation=lambda x: x,
                         padding='SAME',
                         init_type='bilinear',
                         stddev=0.02,
                         bias=0.1
                        ):
        with tf.variable_scope(name_or_scope):
            static_input_shape = x.get_shape().as_list()
            dyn_input_shape = tf.shape(x)

            # extract batch-size like as a symbolic tensor to allow variable size
            batch_size = dyn_input_shape[0]

            if init_type == 'gaussian':
                # Initialization with noise
                # w = tf.get_variable(
                #     'W', [k_h, k_w, n_filters, static_input_shape[3]],
                #     initializer=tf.truncated_normal_initializer(stddev=stddev))
                w = self.weight_variable([k_h, k_w, n_filters, static_input_shape[3]], stddev=stddev )
            else:
                # Initialization with bilinear upscaling
                w = self.get_bilinear_filter([k_h, k_w, n_filters, static_input_shape[3]])

            assert padding in {'SAME', 'VALID'}
            if (padding is 'SAME'):
                out_h = dyn_input_shape[1] * stride_h
                out_w = dyn_input_shape[2] * stride_w
            elif (padding is 'VALID'):
                out_h = (dyn_input_shape[1] - 1) * stride_h + k_h
                out_w = (dyn_input_shape[2] - 1) * stride_w + k_w

            out_shape = tf.pack([batch_size, out_h, out_w, n_filters])

            convt = tf.nn.conv2d_transpose(
                x, w, output_shape=out_shape,
                strides=[1, stride_h, stride_w, 1], padding=padding)

            # b = tf.get_variable(
            #     'b', [n_filters],
            #     initializer=tf.constant_initializer(bias))
            b = self.bias_variable([n_filters], init_bias=bias)

            convt += b
        print 'Shape of ', name_or_scope, ' ', convt.get_shape()

        return activation(convt)

    ## Activation
    def activation(self, input):
        return tf.nn.relu(input, name='activation')

    ## Fully connected layer
    def fc_lyr(self, input, out_num, name, keep_prob=0.5, activation_on=True, batch_norm=False, phase_train=None, debug=True):
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
            print '%s: in_num = %d, out_num = %d ' % (name, in_num, out_num)

        return h_fc_drop


    ## Building inception module V1.0
    def build_inception_module_v1(self, input_layer, input_size, filter_sizes, name, debug):
        with tf.variable_scope(name):
            with tf.variable_scope("pool_proj"):
                # 3x3 stride 1 max pooling and projection
                h_pool = tf.nn.max_pool(input_layer, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                W_pool_proj = self.weight_variable([1, 1, input_size, filter_sizes[0]])
                b_pool_proj = self.bias_variable([filter_sizes[0]])
                h_pool_proj = tf.nn.relu(
                    tf.nn.conv2d(h_pool, W_pool_proj, strides=[1, 1, 1, 1], padding='SAME') + b_pool_proj)

            with tf.variable_scope("1x1"):
                # 1x1 stride 1 convolution
                W_conv1x1 = self.weight_variable([1, 1, input_size, filter_sizes[1]])
                b_conv1x1 = self.bias_variable([filter_sizes[1]])
                h_conv1x1 = tf.nn.relu(
                    tf.nn.conv2d(input_layer, W_conv1x1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1x1)

            with tf.variable_scope("3x3_reduce"):
                # 1x1 convolution reduction and 3x3 stride 1 convolution
                W_conv3x3_reduce = self.weight_variable([1, 1, input_size, filter_sizes[2]])
                b_conv3x3_reduce = self.bias_variable([filter_sizes[2]])
                h_conv3x3_reduce = tf.nn.relu(
                    tf.nn.conv2d(input_layer, W_conv3x3_reduce, strides=[1, 1, 1, 1],
                                 padding='VALID') + b_conv3x3_reduce)

            with tf.variable_scope("3x3"):
                W_conv3x3 = self.weight_variable([3, 3, filter_sizes[2], filter_sizes[3]])
                b_conv3x3 = self.bias_variable([filter_sizes[3]])
                h_conv3x3 = tf.nn.relu(
                    tf.nn.conv2d(h_conv3x3_reduce, W_conv3x3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3x3)

            with tf.variable_scope("5x5_reduce"):
                # 1x1 convolution reduction and 5x5 stride 1 convolution
                W_conv5x5_reduce = self.weight_variable([1, 1, input_size, filter_sizes[4]])
                b_conv5x5_reduce = self.bias_variable([filter_sizes[4]])
                h_conv5x5_reduce = tf.nn.relu(
                    tf.nn.conv2d(input_layer, W_conv5x5_reduce, strides=[1, 1, 1, 1],
                                 padding='VALID') + b_conv5x5_reduce)

            with tf.variable_scope("5x5"):
                W_conv5x5 = self.weight_variable([5, 5, filter_sizes[4], filter_sizes[5]])
                b_conv5x5 = self.bias_variable([filter_sizes[5]])
                h_conv5x5 = tf.nn.relu(
                    tf.nn.conv2d(h_conv5x5_reduce, W_conv5x5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5x5)

            # concatenation
            h_concat = tf.concat(3, [h_conv1x1, h_conv3x3, h_conv5x5, h_pool_proj])

        if debug:
            h_concat = tf.Print(h_concat, [tf.shape(h_concat)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
        print 'Shape of ', name, ' ', h_concat.get_shape()

        return h_concat