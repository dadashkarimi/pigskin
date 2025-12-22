# local python import
import warnings

# third party
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras.layers import InputSpec
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils import conv_utils


# local imports
import neurite as ne
import neurite_sandbox as nes
import voxelmorph as vxm


class SplitConvIntoBlocks:
    '''
    helper function to split a conv layer into multiple individual blocks
    '''
    bno = 0

    def __init__(self, nfilters, nb_blocks, lrs=None, name='SplitConvIntoBlocks', **kwargs):
        self.nb_blocks = nb_blocks
        self.nfilters = []
        self.total_filters = nfilters
        self.name = name
        self.kwargs = kwargs
        self.lrs = lrs
        self.filt_per_block = nfilters // nb_blocks

        if self.filt_per_block < 0:
            print(f'SplitConvIntoBlocks: not enough filters {nfilters} for blocks {nb_blocks}')
            return

        for i in range(self.nb_blocks - 1):
            self.nfilters.append(self.filt_per_block)

        # if not evenly divisible put all the remaining filters in the last block
        self.nfilters.append(nfilters - (self.filt_per_block * (nb_blocks - 1)))

    def __call__(self, input_tensor):
        if self.filt_per_block < 0:
            return

        bno = SplitConvIntoBlocks.bno
        ndims = len(input_tensor.get_shape().as_list()) - 2
        Conv = getattr(KL, 'Conv%dD' % ndims)

        tensors = []
        for i in range(self.nb_blocks):
            layer = Conv(self.nfilters[i], name=f'{self.name}{bno}_subconv{i}', **self.kwargs)
            tensors.append(layer(input_tensor))

        if self.lrs is not None:    # give each layer a different learning rate
            tensors2 = []
            for lno, tensor in enumerate(tensors):
                tensor = KL.Lambda(lambda x: x[0] * x[1], name=f'mul{bno}_subconv{i}')(
                    [tensor, tf.cast(self.lrs[lno], tf.float32)])
                tensors2.append(tensor)

            tensors = tensors2

        SplitConvIntoBlocks.bno += 1
        output_tensor = KL.Concatenate(axis=-1)(tensors) if self.nb_blocks > 1 else tensors[0]
        return output_tensor


def block_conv_layer(nb_blocks=4, lrs=None):
    def conv_block(nfilters, **kwargs):
        return SplitConvIntoBlocks(nfilters, nb_blocks, lrs=lrs, **kwargs)

    return conv_block


class Select_Conv(Layer):
    """
    Sandbox experimental
    Convolution layer with selected output
    """

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Select_Conv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        # self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = input_shape[0]
        channel_axis = -1

        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):
        filter_sel = inputs[1]
        inputs = inputs[0]

        print('warning -- selecting first filter_sel. so each batch gets one sel')
        kernel = tf.gather(self.kernel, filter_sel[0], axis=-1)  # issue here: this is batch-ing.

        if self.rank == 1:
            outputs = K.conv1d(
                inputs,
                kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format="channels_last",
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                kernel,
                strides=self.strides,
                padding=self.padding,
                data_format="channels_last",
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            outputs = K.conv3d(
                inputs,
                kernel,
                strides=self.strides,
                padding=self.padding,
                data_format="channels_last",
                dilation_rate=self.dilation_rate)

        bias = tf.gather(self.bias, filter_sel[0], axis=0)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                bias,
                data_format="channels_last")

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]  # take out first loss.
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (1,)  # 1 for selection


class SpatiallySparseDense_old(Layer):
    """
    Spatially-Sparse Dense Layer (great name, huh?)

    for new (fixed?) implementation, see neurite.SpatiallySparse_Dense
    """

    def __init__(self,
                 input_shape,
                 output_len,
                 use_bias=False, my_initializer='RandomNormal', **kwargs):
        self.initializer = my_initializer
        self.output_len = output_len
        self.cargs = 0
        self.use_bias = use_bias
        self.orig_input_shape = input_shape  # just the image size
        super(SpatiallySparseDense_old, self).__init__(**kwargs)

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.
        self.mult = self.add_weight(name='mult-kernel',
                                    shape=(np.prod(self.orig_input_shape),
                                           self.output_len),
                                    initializer=self.initializer,
                                    trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias-kernel',
                                        shape=(self.output_len, ),
                                        initializer=self.initializer,
                                        trainable=True)

        self.sigma_sq = self.add_weight(name='bias-kernel',
                                        shape=(1, ),
                                        initializer=self.initializer,
                                        trainable=True)

        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, args):

        if not isinstance(args, (list, tuple)):
            args = [args]
        self.cargs = len(args)

        # flatten
        if len(args) == 2:  # input
            mult = K.expand_dims(self.mult, 0)

            data, mask = args

            a_fact = int(data.get_shape().as_list()[-1] / mask.get_shape().as_list()[-1])
            mask = K.repeat_elements(mask, a_fact, -1)

            data_flat = K.expand_dims(K.batch_flatten(data), -1)
            mask_flat = K.expand_dims(K.batch_flatten(mask), -1)

            # get the subset of weights that matter.
            # TODO: it might be faster to *extract* the rows, but im not sure.
            W0 = mask_flat * mult
            W0T = K.permute_dimensions(W0, [0, 2, 1])

            # x = inv(W0T * W0 + sigma_sq^I) * W0T * y + b
            seye = self.sigma_sq * K.expand_dims(K.eye(self.output_len), 0)
            pre = tf.linalg.inv(K.batch_dot(W0T, W0) + seye)
            m = K.batch_dot(W0T, data_flat)
            res = K.batch_dot(pre, m)

            if self.use_bias:
                res += K.expand_dims(K.expand_dims(self.bias, 0), -1)

        else:
            # self.output_shape =

            data = args[0]
            shape = K.shape(data)

            data = K.batch_flatten(data)

            if self.use_bias:
                data -= self.bias

            res = tf.matmul(data, K.permute_dimensions(self.mult, [1, 0]))

            # reshape
            # Here you can mix integers and symbolic elements of `shape`
            pool_shape = tf.stack([shape[0], *self.orig_input_shape])
            res = K.reshape(res, pool_shape)

        return res

    def compute_output_shape(self, input_shape):
        # print(self.cargs, input_shape, self.output_len, self.orig_input_shape)
        if self.cargs == 2:
            return (input_shape[0][0], self.output_len)
        else:
            return (input_shape[0], *self.orig_input_shape)


class LocallyConnected2DMultiChannel(Layer):
    """
    Keras Layer:  apply a set of weights that are location specific to an input
                  image. Input and output can both be multi-channel
       __init__(self, nb_channels)
    nb_channels - # of output channels
    only works for 2D (3D would run out of ram at the moment for reasonable sizes)
    the optional priors initialization  will init the weights based on a prior label map
    """

    def __init__(self, nb_channels, priors=None, **kwargs):
        self.nb_channels = nb_channels
        self.priors = priors

        super(LocallyConnected2DMultiChannel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=input_shape[1:] + (self.nb_channels),
            initializer='RandomNormal',
            name='LCkernel')
        self.bias = self.add_weight(
            shape=input_shape[1:],
            initializer='RandomNormal',
            name='LCbias')
        super(LocallyConnected2DMultiChannel, self).build(input_shape)
        if False and (self.priors is not None):  # disabled
            # assert self.priors.shape[0:2] == input_shape[1:3], 'priors shape ' + priors.shape + '
            # is not equal to input shape ' + input_shape
            wts = np.random.randn(*self.kernel.shape)
            wts /= np.prod(wts.shape)
            bias = np.random.randn(*self.bias.shape)
            bias /= np.prod(bias.shape)
            for x in range(self.priors.shape[0]):
                for y in range(self.priors.shape[1]):
                    for label1 in range(self.priors.shape[2]):
                        for label2 in range(self.nb_channels):
                            wts[x][y][label1][label2] = max(1e-6, self.priors[x][y][label2])
            self.set_weights([wts, bias])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'nb_channels': self.nb_channels,
            #            'priors' : self.priors,
        })
        return config

    def call(self, x):
        for outfilt in range(self.nb_channels):
            f = self.kernel[..., outfilt]
            xch = tf.einsum('...ijk,...ijk->...ij', x, self.kernel[..., outfilt])
            xch += self.bias[..., outfilt]
            if outfilt == 0:
                xout = xch[..., tf.newaxis]
            else:
                xout = tf.concat((xout, xch[..., tf.newaxis]), axis=-1)

        return xout

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:-1] + (self.nb_channels,)
        return output_shape


class SpatiallySparseLocallyConnected3D(Layer):
    """
    code based on LocallyConnected2D from keras layers:
    https://github.com/keras-team/keras/blob/master/keras/layers/local.py

    Locally-connected layer for 2D inputs.
    The `LocallyConnected2D` layer works similarly
    to the `Conv2D` layer, except that weights are unshared,
    that is, a different set of filters is applied at each
    different patch of the input.
    # Examples
    ```python
        # apply a 3x3 unshared weights convolution with 64 output filters on a 32x32 image
        # with `data_format="channels_last"`:
        model = Sequential()
        model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
        # now model.output_shape == (None, 30, 30, 64)
        # notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64 parameters
        # add a 3x3 unshared weights convolution on top, with 32 output filters:
        model.add(LocallyConnected2D(32, (3, 3)))
        # now model.output_shape == (None, 28, 28, 32)
    ```
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        padding: Currently only support `"valid"` (case-insensitive).
            `"same"` will be supported in future.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    # from tensorflow.keras.legacy import interfaces
    from tensorflow.python.keras.utils import conv_utils

    # @interfaces.legacy_conv3d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(SpatiallySparseLocallyConnected3D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, 3, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if self.padding != 'valid':
            raise ValueError('Invalid border mode for LocallyConnected3D '
                             '(only "valid" is supported): ' + padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=5)

        self.build_state = 'forward'

    def build(self, input_shape):

        self.orig_input_shape = input_shape

        if self.data_format == 'channels_last':
            input_row, input_col, input_z = input_shape[1:-1]
            input_filter = input_shape[4]
        else:
            input_row, input_col, input_z = input_shape[2:]
            input_filter = input_shape[1]
        self.orig_input_filter = input_filter
        if input_row is None or input_col is None:
            raise ValueError('The spatial dimensions of the inputs to '
                             ' a LocallyConnected3D layer '
                             'should be fully-defined, but layer received '
                             'the inputs shape ' + str(input_shape))
        output_row = conv_utils.conv_output_length(input_row, self.kernel_size[0],
                                                   self.padding, self.strides[0])
        output_col = conv_utils.conv_output_length(input_col, self.kernel_size[1],
                                                   self.padding, self.strides[1])
        output_z = conv_utils.conv_output_length(input_z, self.kernel_size[2],
                                                 self.padding, self.strides[2])
        self.output_row = output_row
        self.output_col = output_col
        self.output_z = output_z
        self.kernel_shape = (output_row * output_col * output_z,
                             self.kernel_size[0] *
                             self.kernel_size[1] *
                             self.kernel_size[2] * input_filter,
                             self.filters)
        self.kernel = self.add_weight(shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(output_row, output_col, output_z, self.filters),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        if self.data_format == 'channels_first':
            self.input_spec = InputSpec(ndim=5, axes={1: input_filter})
        else:
            self.input_spec = InputSpec(ndim=5, axes={-1: input_filter})
        self.built = True

    def _switch_build_state(self):
        if self.build_state == 'forward':
            self.build_state = 'backward'
            if self.data_format == 'channels_first':
                self.input_spec = InputSpec(ndim=5)
            else:
                self.input_spec = InputSpec(ndim=5)

        else:
            self.build_state = 'forward'
            if self.data_format == 'channels_first':
                self.input_spec = InputSpec(ndim=5, axes={1: self.orig_input_filter})
            else:
                self.input_spec = InputSpec(ndim=5, axes={-1: self.orig_input_filter})

    def compute_output_shape(self, input_shape):
        if self.build_state == 'forward':
            if self.data_format == 'channels_first':
                rows = input_shape[2]
                cols = input_shape[3]
                z = input_shape[4]
            elif self.data_format == 'channels_last':
                rows = input_shape[1]
                cols = input_shape[2]
                z = input_shape[3]

            rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                                 self.padding, self.strides[0])
            cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                                 self.padding, self.strides[1])
            z = conv_utils.conv_output_length(z, self.kernel_size[2],
                                              self.padding, self.strides[2])

            if self.data_format == 'channels_first':
                return (input_shape[0], self.filters, rows, cols, z)
            elif self.data_format == 'channels_last':
                return (input_shape[0], rows, cols, z, self.filters)
        else:
            assert self.build_state == 'backward'
            return self.orig_input_shape

    def call(self, inputs):
        if self.build_state == 'forward':
            output = self.local_conv3d(inputs,
                                       self.kernel,
                                       self.kernel_size,
                                       self.strides,
                                       (self.output_row, self.output_col, self.output_z),
                                       self.data_format)

            if self.use_bias:
                output = K.bias_add(output, self.bias,
                                    data_format=self.data_format)

            output = self.activation(output)
            return output

        else:
            assert self.build_state == 'backward'

            if self.use_bias:
                inputs = K.bias_add(inputs, - self.bias,
                                    data_format=self.data_format)

            output = self.local_unconv3d(inputs,
                                         self.kernel,
                                         self.kernel_size,
                                         self.strides,
                                         (self.output_row, self.output_col, self.output_z),
                                         self.data_format)

            output = self.activation(output)
            return output

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(
            SpatiallySparseLocallyConnected3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def local_conv3d(self, inputs, kernel, kernel_size, strides, output_shape, data_format=None):
        """Apply 2D conv with un-shared weights.
        # Arguments
            inputs: 4D tensor with shape:
                    (batch_size, filters, new_rows, new_cols)
                    if data_format='channels_first'
                    or 4D tensor with shape:
                    (batch_size, new_rows, new_cols, filters)
                    if data_format='channels_last'.
            kernel: the unshared weight for convolution,
                    with shape (output_items, feature_dim, filters)
            kernel_size: a tuple of 2 integers, specifying the
                        width and height of the 2D convolution window.
            strides: a tuple of 2 integers, specifying the strides
                    of the convolution along the width and height.
            output_shape: a tuple with (output_row, output_col)
            data_format: the data format, channels_first or channels_last
        # Returns
            A 4d tensor with shape:
            (batch_size, filters, new_rows, new_cols)
            if data_format='channels_first'
            or 4D tensor with shape:
            (batch_size, new_rows, new_cols, filters)
            if data_format='channels_last'.
        # Raises
            ValueError: if `data_format` is neither
                        `channels_last` or `channels_first`.
        """
        if data_format is None:
            data_format = K.image_data_format()
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format: ' + str(data_format))

        stride_row, stride_col, stride_z = strides
        output_row, output_col, output_z = output_shape
        kernel_shape = K.int_shape(kernel)
        _, feature_dim, filters = kernel_shape

        xs = []
        for i in range(output_row):
            for j in range(output_col):
                for k in range(output_z):
                    slice_row = slice(i * stride_row,
                                      i * stride_row + kernel_size[0])
                    slice_col = slice(j * stride_col,
                                      j * stride_col + kernel_size[1])
                    slice_z = slice(k * stride_z,
                                    k * stride_z + kernel_size[2])

                    if data_format == 'channels_first':
                        xs.append(K.reshape(inputs[:, :, slice_row, slice_col, slice_z],
                                            (1, -1, feature_dim)))
                    else:
                        xs.append(K.reshape(inputs[:, slice_row, slice_col, slice_z, :],
                                            (1, -1, feature_dim)))

        x_aggregate = K.concatenate(xs, axis=0)
        output = K.batch_dot(x_aggregate, kernel)
        output = K.reshape(output,
                           (output_row, output_col, output_z, -1, filters))

        if data_format == 'channels_first':
            output = K.permute_dimensions(output, (3, 4, 0, 1, 2))
        else:
            output = K.permute_dimensions(output, (3, 0, 1, 2, 4))
        return output

    def local_unconv3d(self, inputs, kernel, kernel_size, strides, output_shape, data_format=None):
        """
        Apply 2D un-conv with un-shared weights.
        """
        if data_format is None:
            data_format = K.image_data_format()
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format: ' + str(data_format))

        stride_row, stride_col, stride_z = strides
        output_row, output_col, output_z = output_shape  # self.orig_input_shape[1:-1]
        kernel_shape = K.int_shape(kernel)
        _, feature_dim, filters = kernel_shape

        bs = K.shape(inputs)[0]

        # TODO:
        # - get the input and permute and reshape from [?, 5, 5, 5, 3] to [125, ?, 3]
        # - compute pseudo-inverse of W (kernel). go from [125, 11^3*32, 3] tp [125, 3, 11^3*32]
        # - batch_dot invW with reshaped input to get [125, ?, 11^3*32]
        # - merge the 125 batches into a volume by adding

        if data_format == 'channels_first':
            inputs = K.reshape(inputs, [-1, K.prod(K.shape(inputs)[2:5]), K.shape(inputs)[1]])
        else:
            inputs = K.reshape(inputs, [-1, K.prod(K.shape(inputs)[1:4]), K.shape(inputs)[4]])

        inputs = K.permute_dimensions(inputs, [1, 0, 2])

        W = self.kernel  # 125, 11^3^3, 3
        # compute pseudo-inverse
        Wt = K.permute_dimensions(W, [0, 2, 1])
        wtwinv = tf.matrix_inverse(K.batch_dot(Wt, W))
        pinv = K.batch_dot(wtwinv, Wt)

        outputs_tmp = K.batch_dot(inputs, pinv)  # 125, ?, 11^3*32
        outputs_tmp = K.permute_dimensions(outputs_tmp, [0, 2, 1])  # 125, 11^3*32, ?
        # s = [W.shape[0], *self.kernel_size, self.orig_input_filter, bs]
        # outputs_tmp = K.reshape(outputs_tmp, s)

        shp = (*self.orig_input_shape[1:], bs)
        output = K.zeros(shape=shp, dtype='float32')  # X x Y x Z x filters x bs
        output_tot = K.zeros(shape=shp, dtype='float32') + K.epsilon()

        # go through patches and add them up
        pi = -1
        for i in range(output_row):
            for j in range(output_col):
                for k in range(output_z):
                    slice_row = np.arange(i * stride_row,
                                          i * stride_row + kernel_size[0])
                    slice_col = np.arange(j * stride_col,
                                          j * stride_col + kernel_size[1])
                    slice_z = np.arange(k * stride_z,
                                        k * stride_z + kernel_size[2])
                    slice_filt = np.arange(0, self.orig_input_shape[-1])  # channels-last only
                    pi += 1

                    if data_format == 'channels_last':
                        indices = tf.meshgrid(slice_row, slice_col, slice_z, slice_filt)
                        indices = tf.cast(tf.stack([K.flatten(f)
                                                    for f in indices], axis=1), tf.int32)
                        indices = K.expand_dims(indices, 0)  # 1 x nb_indices(patch_size) x 4

                        u = K.expand_dims(outputs_tmp[pi, :, :], 0)
                        z = tf.scatter_nd(indices, u, K.shape(output))
                        output += z
                        output_tot += tf.scatter_nd(indices, u * 0 + 1, K.shape(output))

                        # output[:, :, slice_row, slice_col, slice_z] += outputs_tmp[pi, :]
                        # output_tot[:, :, slice_row, slice_col, slice_z] += 1
                        # # xs.append(K.reshape(inputs[:, :, slice_row, slice_col, slice_z],
                        #                 # (1, -1, feature_dim)))
                    else:
                        raise ValueError('unimplemented')
                        pass

                        # output[:, slice_row, slice_col, slice_z, :] += outputs_tmp[pi, :]
                        # output_tot[:, slice_row, slice_col, slice_z, :] += 1
                        # # xs.append(K.reshape(inputs[:, slice_row, slice_col, slice_z, :],
                        #                 # (1, -1, feature_dim)))

        output = output / output_tot
        output = K.permute_dimensions(output, [4, 0, 1, 2, 3])            # X x Y x Z x filters x bs

        print(output)

        # if data_format == 'channels_first':
        #     output = K.permute_dimensions(output, (3, 4, 0, 1, 2))
        # else:
        #     output = K.permute_dimensions(output, (3, 0, 1, 2, 4))
        return output


class LatentStack(Layer):
    """
    Based on MeanStream

    Maintain a stack of latest values to come through this layer, and return that stack.
    Useful for maintaining the last <stack_len> latent-space entries, for example.
    """

    def __init__(self, batch_size=16, stack_len=1000, **kwargs):
        # batch_size is fixed!
        self.batch_size_int = batch_size
        self.batch_size = K.variable(batch_size, dtype='float32')
        self.stack_len = stack_len
        super(LatentStack, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create stack
        # This is a weight matrix because just maintaining variables
        # don't get saved with the model, and we'd like
        # to have these numbers saved when we save the model.
        # But we need to make sure that the weights are untrainable.
        self.latent_stack = self.add_weight(name='mean',
                                            shape=[self.stack_len] + list(input_shape[1:]),
                                            initializer='zeros',
                                            trainable=False)

        super(LatentStack, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # get new mean and count
        new_stack = tf.concat([self.latent_stack[self.batch_size_int:, ...], x], 0)

        # update op
        updates = [(self.latent_stack, new_stack)]
        self.add_update(updates, x)

        return self.latent_stack

    def compute_output_shape(self, input_shape):
        return tuple([self.stack_len] + list(input_shape[1:]))


class GaussianBlur(Layer):
    """
    Keras Layer: Gaussian Blurring

    Blur tensor by applying a Gaussian kernel. Sigma should be a scalar or list of scalars.
    """

    def __init__(self, sigma, **kwargs):
        self.sigma = sigma
        super(GaussianBlur, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GaussianBlur, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        blur_fn = lambda x: nes.utils.gaussian_smoothing(x, self.sigma)
        return tf.map_fn(blur_fn, x, dtype='float32')

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sigma': self.sigma,
        })
        return config


def flash_forward(PD, T1, T2star, alpha, TR, TE):
    if T2star is not None:
        PD *= tf.math.exp(-TE / (tf.math.abs(T2star)+.01))
    E1 = tf.math.exp(-TR / (tf.math.abs(T1)+.01)) 
    S = PD * tf.math.sin(alpha) * (1 - E1) / (1 - tf.math.cos(alpha) * E1) 
    return S

class FlashForward(Layer):
    """
    Keras Layer: use the steady-state Bloch equations to transform a parameter map (PD/T1/T2*)
    and a set of acquisition paramters (TR/TE/flip in rads) into a synthetic flash image
    inputs should be a list of:
    (shape, 3) input parameter maps
    (3,) input acquisition parameters (TR/TE/flip in rads)
    (shape, 1) B1+ field
    output is (shape,1) flash image
    """

    def __init__(self, **kwargs):
        super(FlashForward, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FlashForward, self).build(input_shape)  # Be sure to call this somewhere!

    def get_config(self):
        config = super().get_config().copy()
        #config.update({'padding': self.padding})
        return config

    def call(self, x):
        synth_images = tf.map_fn(self._synth_single, x, fn_output_signature='float32')

        return synth_images

    def _synth_single(self, x):
        im = x[0]   # 3-frame volume with PD, T1 and T2*
        mr_parms = x[1]   # MRI parameters TR/flip_angle
        B1_plus = x[2]
        fit_T2star = im.get_shape().as_list()[-1] == 3
        if fit_T2star:
            te_list = x[3]    # list of TEs for this multi-frame mef vol
            T2star = (tf.math.abs(im[..., 2]) + .01)[..., tf.newaxis]

        flip_field = tf.squeeze(tf.multiply(B1_plus, mr_parms[1]))  # now it is a map of flip angles
        synth_image = flash_forward(im[..., 0], im[..., 1], None, flip_field, mr_parms[0], None)[..., tf.newaxis]
        if fit_T2star:
            synth_image = synth_image * tf.math.exp(-te_list / T2star)

        return synth_image

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0]
        return output_shape


class Pad(Layer):
    """
    Keras Layer: pad of the input. padding is a list or tuple of integers,
    one per dimension of input and the padding is uniform in each dimension
    """

    def __init__(self, padding, **kwargs):
        self.padding = padding
        super(Pad, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Pad, self).build(input_shape)  # Be sure to call this somewhere!

    def get_config(self):
        config = super().get_config().copy()
        config.update({'padding': self.padding})
        return config

    def call(self, x):
        ndims = len(x.get_shape().as_list()) - 2
        assert ndims == len(self.padding), 'Pad: ndims %d does not match padding length %d' % (
            ndims, len(self.padding))

        plist = [[0, 0], [self.padding[0], self.padding[0]]]
        if ndims > 1:  # 2D
            plist += [[self.padding[1], self.padding[1]]]
        if ndims > 2:  # 3D
            plist += [[self.padding[2], self.padding[2]]]
        plist += [[0, 0]]
        padding = tf.constant(plist)
        xpadded = tf.pad(x, padding, 'REFLECT')
        return xpadded

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0]]
        output_shape += [int(input_shape[1:-1][f] + self.padding[f])
                         for f in range(len(self.padding))]
        output_shape += [input_shape[-1]]
        return output_shape


class Patches2D(Layer):
    """
    Keras Layer: compute a set of 2D patches from an input image. The returned patches
    will be (batches, psize, psize, nchannels, npatches)
    """

    def __init__(self, psize=32, stride=16, padding='VALID', **kwargs):
        self.padding = padding
        self.psize = psize
        self.stride = stride
        super(Patches2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Patches2D, self).build(input_shape)  # Be sure to call this somewhere!

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'psize': self.psize,
                'stride': self.stride,
                'padding': self.padding,
            })
        return config

    def call(self, x):
        ext_patches = lambda x: nes.utils.extract_patches_from_image_2D(
            x, psize=self.psize, stride=self.stride, padding=self.padding)
        xpatches = tf.map_fn(ext_patches, x, dtype='float32')
        self.out_shape = xpatches.get_shape().as_list()[1:]
        return xpatches

    def compute_output_shape(self, input_shape):
        output_shape = self.out_shape
        return output_shape


class ImageFromPatches2D(Layer):
    """
    Keras Layer:
      inverse of of Patches2D (takes a bunch of patches and recons and image from them
      including accounting for overlap in adjacent patches.  Note that image_size and stride
      must match what is given to Patches2D
    """

    def __init__(self, image_size, stride, **kwargs):
        self.image_size = image_size
        self.stride = stride
        super(ImageFromPatches2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ImageFromPatches2D, self).build(input_shape)  # Be sure to call this somewhere!

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'image_size': self.image_size,
                'stride': self.stride,
            })
        return config

    def call(self, x):
        recon_image = lambda x: nes.utils.patches_to_image_2D(
            x, self.image_size, strides=(self.stride, self.stride))
        xrecon = tf.map_fn(recon_image, x, dtype='float32')
        self.out_shape = xrecon.get_shape().as_list()[1:]
        return xrecon

    def compute_output_shape(self, input_shape):
        output_shape = self.out_shape
        return output_shape


class Repeat(Layer):
    """
    Keras Layer: Repeat Tensor

    TODO: lots to do, basically in  supporting the various aspects of tf.repeat
    """

    def __init__(self, repeats, axis=None, **kwargs):
        self.repeats = repeats
        self.axis = axis
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        x_repeat = tf.repeat(x, self.repeats, self.axis)
        # need to force reshape to avoid None
        shape = tf.concat([tf.shape(x)[:-1], [x.shape[-1] * self.repeats]], 0)
        return tf.reshape(x_repeat, shape)

    def compute_output_shape(self, input_shape):
        # assumes axis is fixed and given for now
        lst_output_shape = list(input_shape)
        lst_output_shape[self.axis] = self.repeats
        return tuple(lst_output_shape)


class GeneralisedLogistic(Layer):
    """
    Keras Layer: apply a generalized logistic function to the intensities
                 of the input image

    # Generalised logistic function
    # y = A + (K-A) / (C + Q exp(-Bt))**(1/v)
    # A is lower asymptote (0)
    # K is upper asymptote (1)
    # B is growth rate
    # C is typically 1
    # --> 1 / (exp(-B t)**(1/v)

    """

    def __init__(self, **kwargs):
        super(GeneralisedLogistic, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GeneralisedLogistic, self).build(input_shape)  # Be sure to call this somewhere!

    def get_config(self):
        config = super().get_config().copy()
        return config

    def call(self, x):
        im = x[0]
        parms = x[1]
        sigmoid_images = tf.map_fn(self._rescale_single, x, fn_output_signature='float32')
        return sigmoid_images

    def _rescale_single(self, x):
        im = x[0]
        parms = x[1]
        B = parms[0]
        v = parms[1]
        sigmoid = 1 / (1 + tf.exp(-B * im) ** (1 / v))
        return sigmoid

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        return input_shape[0:-1] + (1, )


class SegmentWithBasis(Layer):
    """
    Keras Layer: segmentation with a user-specified basis set
    """

    def __init__(self, basis_vols, nb_labels=2, **kwargs):
        self.basis_vols = basis_vols
        self.basis_tensors = [tf.convert_to_tensor(bvol) for bvol in basis_vols]
        self.nb_labels = nb_labels
        conv_filt = np.zeros((*basis_vols[0].shape, len(basis_vols), nb_labels))
        self.conv_filt = tf.convert_to_tensor(basis_vols, dtype=tf.float32)
        super(SegmentWithBasis, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SegmentWithBasis, self).build(input_shape)  # Be sure to call this somewhere!

    def get_config(self):
        config = super().get_config().copy()
        config.update({'basis_vols': self.basis_vols})
        return config

    def call(self, xinp):
        ndims = len(xinp.get_shape().as_list()) - 2
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)
        strides = [1] * (ndims + 2)
        xout = conv_fn(xinp, self.conv_filt, strides, 'SAME')
        return xout

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.nb_labels
        return output_shape


class GaussianNoise(ne.layers.GaussianNoise):
    def __init__(self, *args, **kwargs):
        warnings.warn('nes.layers.GaussianNoise is deprecated in favor of ne.layers.GaussianNoise '
                      'and will be removed in the future')
        super().__init__(*args, **kwargs)


class VarianceStream(Layer):
    """
    Keras Layer: compute and accumulate the variance with a forgetting factor
    """

    def __init__(self, forgetting_factor=0.99, **kwargs):
        self.forgetting_factor = forgetting_factor
        super(VarianceStream, self).__init__(**kwargs)

    def build(self, input_shape):
        subject_batch_shape, _ = input_shape
        self.variance = self.add_weight(name='variance',
                                        shape=subject_batch_shape[1:],
                                        initializer='zeros',
                                        trainable=False)
        super(VarianceStream, self).build(input_shape)

    def call(self, x, training=None):
        # only update variance in training mode
        if training:
            subject_batch, atlas_mean = x
            # compute the variance for the current batch,
            # the result is a 3D tensor without batch dim
            var_curr = K.mean(K.square(subject_batch - atlas_mean), axis=[0])
            var_new = self.forgetting_factor * self.variance + \
                (1 - self.forgetting_factor) * var_curr
            self.variance.assign(var_new)

        # insert new axis on batch to be compati
        # ble with subject and atlas mean
        return self.variance[tf.newaxis, ...]

    def compute_output_shape(self, input_shape):
        subject_batch_shape, _ = input_shape
        return (1, *subject_batch_shape[1:])


class LossEndPoint(Layer):
    """
    Keras Layer: the loss end point that applies loss function to input tensors
    """

    def __init__(self, loss_fn=None, name=None, metric_fn=None, metric_name=None):
        self.loss_fn = loss_fn

        if name is None:
            name = 'lep'

        if isinstance(metric_fn, (list, tuple)):
            self.metric_fn = metric_fn
            self.metric_name = metric_name
        else:
            if metric_fn is not None:
                self.metric_fn = [metric_fn]
                self.metric_name = [metric_name]
            else:
                self.metric_fn = None
                self.metric_name = None

        super(LossEndPoint, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        if self.loss_fn is not None:
            loss = self.loss_fn(*inputs)
        else:
            loss = 0

        if self.metric_fn is not None:
            for m, metric_f in enumerate(self.metric_fn):
                self.add_metric(metric_f(*inputs), name=self.metric_name[m])

        return K.mean(loss)

    def compute_output_shape(self, input_shape):
        return ()


# functional interface of LossEndPoint layer
def create_loss_end(x, loss_fn=None, **kwargs):
    if not isinstance(x, (list, tuple)):
        x = [x]
    return LossEndPoint(loss_fn, **kwargs)(x)


class AddWithDecayingWeight(Layer):
    """
    add two tensors with the weight of the second one decaying over time:

    tout = t1 + t2 * wt
    wt = wt * decay_rate
    """

    def __init__(self, init_wt=.5, decay_rate=.999, **kwargs):
        self.init_wt = init_wt
        self.decay_rate = float(decay_rate)
        super(AddWithDecayingWeight, self).__init__(**kwargs)

    def build(self, input_shape):
        self.wt = self.add_weight(name='add_wt',
                                  shape=[1],
                                  initializer='zeros',
                                  trainable=False)

        self.wt.assign([self.init_wt])
        super(AddWithDecayingWeight, self).build(input_shape)

    def call(self, x, training=None):
        output_tensor = x[0] + (self.wt[0]) * x[1]
        if training:
            new_wt = self.wt[0] * self.decay_rate   # apply exponential decay
            self.wt.assign(new_wt)

        return output_tensor

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.function
def _pos_encoding2D(inshape, npos, pad_size=0):
    pe = np.zeros(tuple(inshape) + (2 * npos,))

    for pno in range(npos):
        # x axis / horizontal
        wave_len = inshape[1] / (2 ** pno)
        res = 2 * np.pi / wave_len
        pex_1d = np.cos(np.arange(inshape[1]) * res)
        pex_2d = np.repeat(pex_1d[np.newaxis, ...], inshape[0], axis=0)
        pe[..., pno] = pex_2d

        # y axis / vertical
        wave_len = inshape[0] / (2 ** pno - 0.5)  # -0.5 to avoid same value on two poles
        res = 2 * np.pi / wave_len
        pey_1d = np.cos(np.arange(inshape[0]) * res)
        pey_2d = np.repeat(pey_1d[..., np.newaxis], inshape[1], axis=1)
        pe[..., npos + pno] = pey_2d

    if pad_size > 0:
        pe = nes.utils.pad_2d_image_spherically(pe, pad_size, input_no_batch_dim=True)

    return tf.convert_to_tensor(pe, tf.float32)


class ConcatWithPositionalEncoding(KL.Layer):
    """
    concatenate a set of positional encoding tensors to an existing one
    """
    def __init__(self, npos, pad_size=0, **kwargs):
        self.npos = npos
        self.pad_size = pad_size
        super(ConcatWithPositionalEncoding, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'npos': self.npos,
        })
        return config

    def build(self, inshape):  # create the pe tensor
        input_shape = inshape.as_list()[1:-1]
        if self.pad_size > 0:
            input_shape = [x - 2 * self.pad_size for x in input_shape]
        self.pe = _pos_encoding2D(input_shape, self.npos, self.pad_size)
        super().build(inshape)

    def call(self, x):         # concat of x and self.pe
        conc_fn = lambda x: tf.concat([x, self.pe], axis=-1)
        out = tf.map_fn(conc_fn, x)
        return out

    def compute_output_shape(self, input_shape, **kwargs):
        return input_shape[0:-1] + (input_shape[-1] + 2 * self.npos,)
