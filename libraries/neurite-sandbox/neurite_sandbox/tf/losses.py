"""
tf/keras experimental losses for neurite.

Please respect pep8 rules, thorough naming strategies (even if cumbersome).
Please also add your github username in the documentation of the class/function.
"""


# core
import os
import sys
import warnings

# third party
import numpy as np
import scipy
import tensorflow.keras.backend as K
import tensorflow.keras.losses
import tensorflow.keras
import tensorflow as tf

# locals
import neurite as ne
from . import utils


class VAE:
    """
    Losses for variational auto-encoders
    """

    def __init__(self, log_sigma_out=None, mu_out=None):
        self.log_sigma_out = log_sigma_out
        self.mu_out = mu_out
        self.axis = None
        if mu_out is not None:
            self.axis = [*range(1, len(mu_out.get_shape()))]

    def mse(self, y_true, y_pred):
        return K.mean(tensorflow.keras.losses.mean_squared_error(y_true, y_pred))

    def kl_log_sigma(self, _, y_pred):
        """
        kl_log_sigma terms of the KL divergence

        y_pred should be log_sigma_out
        """
        if self.axis is None:
            self.axis = [*range(1, len(y_pred.get_shape()))]
        kl_sigma_out = 0.5 * K.sum(K.exp(y_pred) - 1. - y_pred, axis=self.axis)
        return kl_sigma_out

    def kl_mu(self, y_true, y_pred):
        """
        kl_mu terms of the KL divergence

        y_pred should be mu_out
        """
        if self.axis is None:
            self.axis = [*range(1, len(y_pred.get_shape()))]
        kl_mu_out = 0.5 * K.sum(K.square(y_pred - y_true), axis=self.axis)
        return kl_mu_out

    def complete_loss(self, y_true, y_pred):
        """
        Using implementation from here as initial guidance:

        https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/

        To balance the terms, there is a natural term that should be computed,
        which is not covered below

        Comments from blog post:
        Calculate loss = reconstruction loss + KL loss for each data in minibatch
        """
        # E[log P(X|z)]
        recon = K.sum(
            tensorflow.keras.losses.mean_squared_error(y_true, y_pred))

        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        term_kl = K.exp(self.log_sigma_out) + \
            K.square(self.mu_out) - 1. - self.log_sigma_out
        kl = 0.5 * K.sum(term_kl, axis=self.axis)

        return recon + kl


class Sparse:

    def __init__(self, mask_tensor=None):
        self.mask_tensor = mask_tensor

    def mse(self, y_true, y_pred):
        y_true, y_pred, true_mask, sparsity = self._prep(y_true, y_pred)
        # returns nb_batch x 1
        return K.mean(true_mask * K.square(y_pred - y_true), 1) / sparsity

    def mae(self, y_true, y_pred):
        y_true, y_pred, true_mask, sparsity = self._prep(y_true, y_pred)
        # returns nb_batch x 1
        return K.mean(true_mask * K.abs(y_pred - y_true), 1) / sparsity

    def _prep(self, y_true, y_pred):
        # get sparsity mask
        if self.mask_tensor is None:
            # assume stacked output
            true_mask = y_true[..., 1]
            y_true = y_true[..., 0]
            y_pred = y_pred[..., 0]
        else:
            true_mask = self.mask_tensor

        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)
        true_mask = K.batch_flatten(true_mask)

        # sparsity of mask = sum(non_zero)/numel
        sparsity = K.mean(true_mask, 1)
        return (y_true, y_pred, true_mask, sparsity)


class DynamicallyWeightedDice:
    """
    Dice loss, weighted dynamically by a weight tensor
    """

    def __init__(self, weight_tensor, laplace_smoothing=0.1):
        self.weights = weight_tensor
        self.laplace_smoothing = laplace_smoothing

    def dice(self, y_true, y_pred):
        """
        IMPORTANT NOTE does no error checking, etc. 
        ys and self.weights are expected to be of size [B, ..., C]
        ys are expected to be in [0, 1] and add up to 1 appropriately
        """

        # reshape to [batch_size, nb_voxels, nb_labels]
        y_true = ne.utils.batch_channel_flatten(y_true)
        y_pred = ne.utils.batch_channel_flatten(y_pred)

        w = ne.utils.batch_channel_flatten(self.weights)

        # compute dice for each entry in batch.
        # dice will now be [batch_size, nb_labels]
        top = 2 * K.sum(w * y_true * y_pred, 1)
        bottom = K.sum(w * K.square(y_true), 1) + K.sum(w * K.square(y_pred), 1)

        # compute Dice
        eps = self.laplace_smoothing
        return (top + eps) / (bottom + eps)


class WGAN_GP:
    """
    based on https://github.com/rarilurelo/keras_improved_wgan/blob/master/wgan_gp.py
    """

    def __init__(self, disc, batch_size=1, lambda_gp=10):
        self.disc = disc
        self.lambda_gp = lambda_gp
        self.batch_size = batch_size

    def loss(self, y_true, y_pred):

        # get the value for the true and fake images
        disc_true = self.disc(y_true)
        disc_pred = self.disc(y_pred)

        # sample a x_hat by sampling along the line between true and pred
        # z = tf.placeholder(tf.float32, shape=[None, 1])
        # shp = y_true.get_shape()[0]
        # WARNING: SHOULD REALLY BE shape=[batch_size, 1] !!!
        # self.batch_size does not work, since it's not None!!!
        alpha = K.random_uniform(shape=[K.shape(y_pred)[0], 1, 1, 1])
        diff = y_pred - y_true
        interp = y_true + alpha * diff

        # take gradient of D(x_hat)
        gradients = K.gradients(self.disc(interp), [interp])[0]
        grad_pen = K.mean(
            K.square(K.sqrt(K.sum(K.square(gradients), axis=1)) - 1))

        # compute loss
        return (K.mean(disc_pred) - K.mean(disc_true)) + self.lambda_gp * grad_pen


class NCCGuhaExperimental:
    """
    Local (over window) normalized cross correlation loss.

    This is a small "fix" to the NCC in vxm.

    Messages from Guha on slack, 10/18/2020:
    i feel like it shouldnt be necessary to compute so many volumes since we are dealing with 
    averages and stuff, and there should be some simplification somewhere
    but i dont see it right now
    equation 6 in our paper is what i was looking at: https://arxiv.org/pdf/1809.05231.pdf
    ask your guy to expand out the quadratics and make sure what Ii did looks right
    oh this is the torch, i can do the keras one
    """

    def __init__(self, win=None, padding='SAME', eps=1e-7):
        self.win = win
        self.eps = eps
        self.padding = padding

    def ncc(self, Ii, Ji):
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(Ii.get_shape().as_list()) - 2
        assert ndims in [
            1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        Ii2 = Ii * Ii
        Ji2 = Ji * Ji
        IiJi = Ii * Ji

        # compute filters
        in_ch = Ji.get_shape().as_list()[-1]
        sum_filt = tf.ones([*self.win, in_ch, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)

        # compute local sums via convolution
        padding = self.padding
        Ii_sum = conv_fn(Ii, sum_filt, strides, padding)
        Ji_sum = conv_fn(Ji, sum_filt, strides, padding)
        Ii2_sum = conv_fn(Ii2, sum_filt, strides, padding)
        Ji2_sum = conv_fn(Ji2, sum_filt, strides, padding)
        IiJi_sum = conv_fn(IiJi, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win) * in_ch
        u_Ii = Ii_sum / win_size
        u_Ji = Ji_sum / win_size

        # New: Compute these additional volumes so that the NCC loss exactly
        # matches paper's formula.
        Ii_u_Ii_sum = conv_fn(Ii * u_Ii, sum_filt, strides, padding)
        Ji_u_Ji_sum = conv_fn(Ji * u_Ji, sum_filt, strides, padding)
        Ii_u_Ji_sum = conv_fn(Ii * u_Ji, sum_filt, strides, padding)
        Ji_u_Ii_sum = conv_fn(Ji * u_Ii, sum_filt, strides, padding)
        u_Ii_u_Ji_sum = conv_fn(u_Ii * u_Ji, sum_filt, strides, padding)
        u_Ii_u_Ii_sum = conv_fn(u_Ii * u_Ii, sum_filt, strides, padding)
        u_Ji_u_Ji_sum = conv_fn(u_Ji * u_Ji, sum_filt, strides, padding)

        cross = IiJi_sum - Ii_u_Ji_sum - Ji_u_Ii_sum + u_Ii_u_Ji_sum
        Ii_var = Ii2_sum - 2 * Ii_u_Ii_sum + u_Ii_u_Ii_sum
        Ji_var = Ji2_sum - 2 * Ji_u_Ji_sum + u_Ji_u_Ji_sum

        cc = (cross * cross + self.eps) / (Ii_var * Ji_var + self.eps)
        return cc

        # return mean cc for each entry in batch
        # return tf.reduce_mean(K.batch_flatten(cc), axis=-1)

    def loss(self, y_true, y_pred):
        return - self.ncc(y_true, y_pred)


# https://gist.github.com/Hvass-Labs/ac4ef0074fd182c6778532ba222d8c37
def total_variation3D(vols, name=None):
    """Calculate and return the Total Variation for one or more vols.

    The total variation is the sum of the absolute differences for neighboring
    pixel-values in the input vols. This measures how much noise is in the vols.

    This can be used as a loss-function during optimization so as to suppress noise
    in vols. If you have a batch of vols, then you should calculate the scalar
    loss-value as the sum: `loss = tf.reduce_sum(tf.image.total_variation(vols))`

    This implements the anisotropic 2-D version of the formula described here:

    https://en.wikipedia.org/wiki/Total_variation_denoising

    Args:
        vols: 5-D Tensor of shape `[batch, width, height, depth, channels]` or
                4-D Tensor of shape `[width, height, depth, channels]`.

        name: A name for the operation (optional).

    Raises:
        ValueError: if vols.shape is not a 4-D or 5-D vector.

    Returns:
        The total variation of `vols`.

        If `vols` was 4-D, return a 1-D float Tensor of shape `[batch]` with the
        total variation for each image in the batch.
        If `vols` was 3-D, return a scalar float with the total variation for that image.
    """

    ndims = vols.get_shape().ndims

    if ndims == 3:
        # The input is a single image with shape [width, height, depth].

        # Calculate the difference of neighboring pixel-values.
        # The vols are shifted one pixel along the height and width by slicing.
        pixel_dif1 = vols[1:, :, :] - vols[:-1, :, :]
        pixel_dif2 = vols[:, 1:, :] - vols[:, :-1, :]
        pixel_dif3 = vols[:, :, 1:] - vols[:, :, :-1]

        # Sum for all axis. (None is an alias for all axis.)
        sum_axis = None
    elif ndims == 4:
        # The input is a single image with shape [width, height, depth, channels].

        # Calculate the difference of neighboring pixel-values.
        # The vols are shifted one pixel along the height and width by slicing.
        pixel_dif1 = vols[1:, :, :, :] - vols[:-1, :, :, :]
        pixel_dif2 = vols[:, 1:, :, :] - vols[:, :-1, :, :]
        pixel_dif3 = vols[:, :, 1:, :] - vols[:, :, :-1, :]

        # Sum for all axis. (None is an alias for all axis.)
        sum_axis = None
    elif ndims == 5:
        # The input is a batch of vols with shape [batch, height, width, channels].

        # Calculate the difference of neighboring pixel-values.
        # The vols are shifted one pixel along the height and width by slicing.
        pixel_dif1 = vols[:, 1:, :, :, :] - vols[:, :-1, :, :, :]
        pixel_dif2 = vols[:, :, 1:, :, :] - vols[:, :, :-1, :, :]
        pixel_dif3 = vols[:, :, :, 1:, :] - vols[:, :, :, :-1, :]

        # Only sum for the last 3 axis.
        # This results in a 1-D tensor with the total variation for each image.
        sum_axis = [1, 2, 3, 4]
    else:
        raise ValueError('\'vols\' must be either '
                         '3 (no channels), 4 (no batch) or 5-dimensional (with batch).')

    # Calculate the total variation by taking the absolute value of the
    # pixel-differences and summing over the appropriate axis.
    tot_var = tf.reduce_sum(tf.math.abs(pixel_dif1), axis=sum_axis) + \
        tf.reduce_sum(tf.math.abs(pixel_dif2), axis=sum_axis) + \
        tf.reduce_sum(tf.math.abs(pixel_dif3), axis=sum_axis)

    return tot_var


def total_variation3D_loss(_, y_pred):
    return total_variation3D(y_pred)


###############################################################################
# decorators
###############################################################################

def crop_tensors_decorator(metric, crop_indices, *args, **kwargs):
    """
    decorator for cropping inputs of a metric

    # TODO expand different ways to crop

    Args:
        metric (function): metric to be decorated, has to have arguments y_true, y_pred, ...
        crop_indices (tf.int): indices for tf.gather inside each element of a batch

    Authors: adalca
    """
    def decorator(y_true, y_pred, *args, **kwargs):
        y_true = ne.utils.batch_gather(y_true, crop_indices)
        y_pred = ne.utils.batch_gather(y_pred, crop_indices)
        return metric(y_true, y_pred, *args, **kwargs)
    return decorator


def channelwise_losses_decorator(frame_list, loss_funcs, loss_weights=None, norm_weights=False):
    """
    loss function that allows a stacked tensor to be separated into lists of frames
    with each list processed by a different loss function (optonally with a different weight)

    Parameters:
        frame_list :   list of lists of frames to process through separate loss. 
                    e.g. frame_list=[0,1]   frame 0 will be passed to loss_funcs[0] and
                                            frame 1 to loss_funcs[1]
                        frame_list=[[0,1],[2]]  frames 0 and 1 will be stacked and passed to
                                                loss_funcs[0] and frame 2 will be 
                                                passed to loss_func[1]

        loss_funcs:    list of loss functions to apply to the frames in frame_list
        loss_weights:  list of weights for each loss functional 
            (optional, defaults to 1 for each loss)
        norm_weights:  divide the total loss by the sum of all loss weights

    Authors: brf2
    """

    assert len(frame_list) == len(loss_funcs), \
        'frame_list (%d) and loss_funcs (%d) must have same len' % (
            len(frame_list), len(loss_funcs))

    if loss_weights is not None:
        assert len(frame_list) == len(loss_weights), \
            'frame_list (%d) and loss weights (%d) must have same len' % \
            (len(frame_list), len(loss_weights))

    def loss(y_true, y_pred):
        total_loss = 0
        total_weight = 0
        for find, frames in enumerate(frame_list):
            if type(frames) is not list:
                frames = [frames]

            # assemble a y_true and y_pred using just the specified frames
            yt = tf.gather(y_true, frames, axis=-1)
            yp = tf.gather(y_pred, frames, axis=-1)

            # compute the loss contribution for this set of frames
            loss_weight = loss_weights[find] if loss_weights is not None else 1
            loss_func = loss_funcs[find]
            total_loss += loss_weight * loss_func(yt, yp)
            total_weight += loss_weight

        if norm_weights:
            total_loss /= total_weight

        return total_loss

    return loss


###############################################################################
# deprecation block
###############################################################################

class MutualInformation(ne.metrics.MutualInformation):
    """
    Deprecate anytime after 12/01/2021

    see ne.metrics.MutualInformation
    """

    def __init__(self, **kwargs):
        warnings.warn('nes.metrics.MutualInformation has moved to ne.metrics.MutualInformation, '
                      'and some of the helper functions have changed slightly.'
                      'This class will be deprecated.')

        super().__init__(self, **kwargs)

    def channelwise_mi(self, x, y):
        self.channelwise(x, y)

    def maps_to_mi(self, x, y):
        self.maps(x, y)


class ExperimentalMatchingVolumes(object):
    """
    Deprecate after 12/01/2021 or figure out what this was...

    Losses for matching segmentation volumes (sums)
    """

    def __init__(self, use_log=True, min_sum=1):
        warnings.warn('Will be deprecated unless we figure out what this was meant to be. '
                      'Not sure what this was meant to be, it\'s one of the original losses')
        self.use_log = use_log
        self.min_sum = min_sum

    def loss(self, y_true, y_pred):  # y_true, y_pred
        def sum_chn(z):
            return K.sum(K.batch_flatten(z), axis=1)

        sum_x = sum_chn(y_true[:, :, :, :, 1])
        sum_y = sum_chn(y_pred[:, :, :, :, 1])

        if self.use_log:
            logsum_x = K.log(K.maximum(sum_x, K.epsilon()))
            logsum_y = K.log(K.maximum(sum_y, K.epsilon()))
            norm_l2 = K.mean(K.abs(logsum_x - logsum_y))

        else:
            norm_l2 = K.mean(K.abs(sum_x - sum_y) /
                             (K.maximum(sum_x, self.min_sum)))

        return norm_l2


class dtrans_loss:
    """
    Losses for narrowband distance transform
    argument to __init__:
    which_loss='mse',            loss function to use
    border=4,                    region in which to compute distance
    far_away_wt=.1,              weight given to voxels that are more than border away
    use_eight_connectivity=True  whether to use 8 or 27 connectivity
    """

    def dtrans_elu(self, dist, border=0, alpha=1):
        ret_func = tf.keras.activations.elu(- (dist - border), alpha=alpha) + alpha
        return ret_func

    def __init__(self, which_loss='mse', border=4, far_away_wt=.1, use_eight_connectivity=True):
        self.which_loss = which_loss
        self.border = border
        self.far_away_wt = far_away_wt
        self.eight_connectivity = use_eight_connectivity
        if self.eight_connectivity:
            sum_filt = np.zeros((3, 3, 3, 1, 1))
            sum_filt[1, 1, 1, 0, 0] = 1
            sum_filt[1, 1, 2, 0, 0] = 1
            sum_filt[2, 1, 1, 0, 0] = 1
            sum_filt[0, 1, 1, 0, 0] = 1
            sum_filt[1, 1, 0, 0, 0] = 1
            sum_filt[1, 2, 1, 0, 0] = 1
            sum_filt[1, 0, 1, 0, 0] = 1
            sum_filt = tf.convert_to_tensor(sum_filt, dtype=tf.float32)
        else:
            sum_filt = tf.ones([3, 3, 3, 1, 1])

        self.sum_filt = sum_filt

    def loss(self, y_true, y_pred):
        # note that in SynthStrip version the synth model was embedded in the stripping
        # model and so the label map was y_pred[...,-1] and needed to be stripped
        pvals_pred = y_pred
        label_map = y_true
        # commented out from strip_loss
        # pvals_true =
        #     tf.map_fn(label_map_to_stripped, label_map, dtype=tf.float32)[...,tf.newaxis]
        pvals_true = label_map

        ndims = len(y_pred.shape) - 2

        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)
        dtrans = tf.zeros(tf.shape(pvals_true))
        prev_outside = pvals_true
        prev_inside = pvals_true
        strides = [1] * (ndims + 2)

        # build approximation to distance transform in a narrow band border voxels wide
        inverse_image = tf.cast(pvals_true == 0, tf.float32)
        for bval in range(1, self.border + 1):
            outside = tf.cast(conv_fn(prev_outside, self.sum_filt, strides, 'SAME') > 0,
                              tf.float32)
            inside = tf.cast(conv_fn(inverse_image, self.sum_filt, strides, 'SAME') == 0,
                             tf.float32)
            inside_border = prev_inside - tf.cast(inside, tf.float32)       # the next ring inside
            outside_border = tf.cast(outside, tf.float32) - prev_outside  # the next ring outside
            prev_outside = tf.cast(outside, tf.float32)
            prev_inside = tf.cast(inside, tf.float32)
            inverse_image = tf.cast(prev_inside == 0, dtype=tf.float32)
            dtrans = tf.add(dtrans, bval * outside_border)
            dtrans = tf.add(dtrans, -bval * inside_border)

        # set everything more than border voxels from the boundary to +/- border+1
        interior = tf.cast(tf.math.logical_and((dtrans == 0), (pvals_true > 0)), tf.float32)
        exterior = tf.cast(tf.math.logical_and((dtrans == 0), (pvals_true <= 0)), tf.float32)
        dtrans = tf.add(dtrans, -(self.border + 1) * interior)
        dtrans = tf.add(dtrans, (self.border + 1) * exterior)

        # downweight locations that are far from boundary
        weight = tf.ones(tf.shape(dtrans))
        far_away = (tf.abs(dtrans) > self.border)
        weight = weight - (1.0 - self.far_away_wt) * tf.cast(far_away, tf.float32)

        if self.which_loss == 'mse':   # only mse and elu supported now
            lmap = tf.reduce_sum(weight * tf.math.squared_difference(pvals_pred, dtrans))
        elif self.which_loss == 'elu':   # not working yet!!!
            diff = tf.subtract(pvals_pred, dtrans)
            lmap = self.dtrans_elu(diff, border=0)  # border already subtracted
        else:
            assert 0, 'invalid loss function %s' % self.which_loss

        lval = tf.math.divide_no_nan(lmap, tf.reduce_sum(weight))

        return lval


def label_coefficient_of_variation(y_true, y_pred):
    ''' 
    compute the mean and variance for each label separately, then compute the coefficient of
    variation (variance / mean**2) for each class separately so that
    the classes are equally weighted regardless of frequency

    returns the coefficient of variation of each class in a (1,nlabels-1) tensor
    the bg label is ignored (which is why it is nlabels-1)

    y_pred should have a ground-truth label map in frame 1 
    '''
    label_map = tf.keras.utils.to_categorical(y_pred[..., 1:2])
    pred_intensity_map = y_pred[..., 0:1]

    inshape = tf.shape(y_true)
    ndim = len(inshape) - 2
    nlabels = tf.shape(label_map)[-1]
    pred_intensity_vec = tf.reshape(pred_intensity_map, [1, -1])    # 1 x nvoxels

    label_vec = tf.reshape(label_map, [-1, nlabels])  # nvoxels x nlabels
    label_sum_sq = tf.linalg.matmul(tf.math.square(pred_intensity_vec), label_vec)  # 1 x nlabels
    pred_label_sum = tf.linalg.matmul(pred_intensity_vec, label_vec)  # 1 x nlabels
    label_nvox = tf.reduce_sum(label_vec, axis=0)        # number of vox in each label
    pred_label_mean = tf.math.divide_no_nan(pred_label_sum, label_nvox)
    pred_label_mean_sq = tf.square(pred_label_mean)
    label_var = tf.math.divide_no_nan(label_sum_sq, label_nvox) - pred_label_mean_sq

    # [:,1:] ignores background voxels (where label is 0). Make sure 0/0 goes to 1 not 0
    eps = sys.float_info.epsilon
    numer = label_var[:, 1:] + eps
    denom = pred_label_mean_sq[:, 1:] + eps
    coef_var = tf.math.divide_no_nan(numer, denom)
    return coef_var


class debug_loss:
    '''
    loss function wrapper to drop into the debugger when the loss
    function exceeds a user specified threshold

    parameters are:
    lfunc        - pointer to loss function to call
    thresh       - loss value threshold (100) 
    use_fv       - bring up a freeview window showing previous and current outputs (True)
    burn_in_epochs - number of epochs to wait before checking threshold (5)
    steps_per_epoch - number of steps in each epoch (50)
    one_per_epoch   - whether disable dropping into the debugger more than once/epoch (True)
    input_stacked - if True will extract the 2nd frame of y_pred to display 
          (this is a way to pass the input images to be visualized)
    use_debugger - drop into the debugger if threshold is exceeded (True)
    '''

    def __init__(self,
                 lfunc,
                 thresh=100,
                 use_fv=True,
                 burn_in_epochs=5,
                 steps_per_epoch=50,
                 one_per_epoch=True,
                 input_stacked=False,
                 use_debugger=True,
                 min_epoch=0):
        self.lfunc = lfunc
        self.use_fv = use_fv
        self.use_debugger = use_debugger
        self.thresh = thresh
        self.y_pred_prev = None
        self.y_true_prev = None
        self.input_stacked = input_stacked
        self.y_inp_prev = None
        self.burn_in_epochs = burn_in_epochs
        self.one_per_epoch = one_per_epoch
        self.steps_per_epoch = steps_per_epoch
        self.steps = 0
        self.epoch = 0
        self.last_epoch = -1

    def loss(self, y_true, y_pred):
        if self.input_stacked:
            y_inp = y_pred[..., 1]
            y_pred_unstacked = y_pred[..., 0:1]
        else:
            y_inp = y_pred
            y_pred_unstacked = y_pred

        lval = self.lfunc(y_true, y_pred_unstacked)
        lval = tf.reduce_mean(lval)

        if tf.math.is_nan(lval) or (
                lval > self.thresh and self.epoch > self.burn_in_epochs and
                (not self.one_per_epoch or self.epoch != self.last_epoch)):

            self.last_epoch = self.epoch
            if self.use_fv:
                import freesurfer as fs
                fv = fs.Freeview(swap_batch_dim=True)
                fv.vol(y_true.numpy(), name='y_true', opts=':colormap=lut')
                fv.vol(self.y_true_prev.numpy(), name='y_true_prev', opts=':colormap=lut')
                if self.input_stacked:  # input image is in last frame
                    fv.vol(y_inp.numpy(), name='input_image')
                    fv.vol(self.y_inp_prev.numpy(), name='prev_input_image')
                fv.vol(y_pred[..., 0:1].numpy(), name='y_pred', opts=':colormap=heat')
                fv.vol(self.y_pred_prev[..., 0:1].numpy(), name='y_pred_prev',
                       opts=':colormap=heat')
                fv.show(title=f'epoch{self.epoch}_step_{self.steps}_lval{lval}')

            if self.use_debugger:
                import pdb as gdb
                gdb.set_trace()

            print(f'lval is {lval}')

        self.y_pred_prev = tf.identity(y_pred)
        if self.input_stacked:  # user has stacked an input volume into the output
            self.y_inp_prev = tf.identity(y_pred[..., 1])

        self.y_true_prev = tf.identity(y_true)
        self.steps += 1
        if self.steps == self.steps_per_epoch:
            self.steps = 0
            self.epoch += 1

        return lval


def centroid_loss(y_true, y_pred):
    ''' compute the mse of the centroids of two volumes '''
    y_true = tf.reshape(y_true, tf.shape(y_true)[0:-1])
    y_pred = tf.reshape(y_pred, tf.shape(y_pred)[0:-1])
    c1 = tf.map_fn(utils.utils.volume_centroid, y_true)
    c2 = tf.map_fn(utils.utils.volume_centroid, y_pred)
    diff = tf.math.squared_difference(c1, c2)
    loss = tf.reduce_mean(diff)

    return loss


def mean_squared_false_error(y_true, y_pred):
    ''' from:
    Wang, S., Liu, W., Wu, J., Cao, L., Meng, Q., Kennedy, P.J.: Training deep neural 
    networks on imbalanced data sets. In: 2016 international joint conference on neural 
    networks (IJCNN). pp. 4368–4374. IEEE (2016)

    compute the mse for each label separately, then average over those so that
    the classes are equally weighted regardless of frequency

    returns the average mse of each class in a (1,nlabels) tensor
    '''
    inshape = tf.shape(y_true)
    ndim = len(inshape) - 2
    nlabels = inshape[-1]
    one_hot = tf.cast(y_true > 0, tf.float32)  # change log probs into probs if given
    error_map = tf.math.squared_difference(y_true, y_pred)
    error_vec = tf.reshape(tf.reduce_sum(error_map, axis=-1), [1, -1])  # 1 x nvoxels

    label_vec = tf.reshape(one_hot, [-1, nlabels])  # nvoxels x nlabels
    numer = tf.linalg.matmul(error_vec, label_vec)  # 1 x nlabels
    denom = tf.reduce_sum(label_vec, axis=0)
    denom = tf.multiply(tf.cast(nlabels, denom.dtype), denom)  # scale so it is per-vox
    denom = tf.clip_by_value(denom, 1e-4, 1e10)  # might be classes that have no vox
    lval = tf.math.divide_no_nan(numer, denom)
    return lval


class DiceNonzero:
    def __init__(self, nlabels, onehot_ytrue=False, weights=None, check_input_limits=False):
        self.nlabels = nlabels
        self.weights = weights
        self.onehot_ytrue = onehot_ytrue
        self.lfunc = ne.losses.Dice(nb_labels=nlabels,
                                    weights=weights,
                                    check_input_limits=check_input_limits).loss

    def loss(self, y_true, y_pred):
        if self.onehot_ytrue:
            y_true = tf.cast(tf.one_hot(y_true, self.nlabels), y_pred.dtype)

        # take mean dice across classes ignorning ones with no labels
        class_counts = K.sum(ne.utils.batch_channel_flatten(y_true), 1)
        lval = self.lfunc(y_true, y_pred)
        lval = tf.reduce_sum(lval) / tf.reduce_sum(tf.clip_by_value(class_counts, 0, 1))
        return lval
