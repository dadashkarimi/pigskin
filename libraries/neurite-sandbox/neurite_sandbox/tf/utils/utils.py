"""
tensorflow/keras utilities for the neuron project

If you use this code, please cite
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation,
CVPR 2018

or for the transformation/interpolation related functions:

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018.

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

# internal python imports
import warnings
import traceback
import math
import re

# third party imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import matplotlib.pyplot as plt

# local imports
import neurite as ne


def blurring_tensor(ndims, nb_input, nb_output, sigma=2):
    '''
    compute a Gaussian blurring tensor that applies to a typical
    network image of shape [batch_size, x, y, nb_input]

    Thus for a 2D image that is (X,Y, nchannels) in shape, the appropriate op
    would be (assuming it should smooth each channel):
    bt = blurring_tensor(2, nchannels, nchannels)
    b = tf.nn.conv2d(tf.cast(tf.convert_to_tensor(im[np.newaxis]),tf.float32), bt, [1]*4, 'SAME')

    If an anisotropic filter is desired a list of sigmas (one per dimension)
    can be given instead of a scalar
    '''

    assert ndims in [2, 3], 'ndims %d not supported by blurring_tensor, must be in [2,3]' % ndims

    if not isinstance(sigma, list):
        sigma = [sigma] * ndims
    blur_kernel = ne.utils.gaussian_kernel(sigma)
    blur_kernel_reshaped = tf.reshape(blur_kernel, blur_kernel.shape.as_list() + [1, 1])
    zeros = tf.zeros((tf.shape(blur_kernel_reshaped)))  # no mixing of channels
    full_blur_kernel = []

    # build a kernel where only the frames that represent the same input
    # and output channel (i==j) are nonzero
    for i in range(nb_input):
        var = []
        for j in range(nb_output):
            if i == j:
                var.append(blur_kernel_reshaped)
            else:
                var.append(zeros)
        full_blur_kernel.append(tf.concat(var, -1))
    full_blur_kernel = tf.concat(full_blur_kernel, -2)
    return full_blur_kernel


def gaussian_smoothing(x, sigmas):
    """
    Blur Tensor of shape [*vol_shape, features] by applying a Gaussian  kernel.

    Important: last channel *must* be number of features! If it's not,
        the algorithm will likely not do what you want

    This function uses a (fixed) separable kernel.

    TODO: would be nice to have the ability to change the filter values
        without needing to re-create a tf graph.

    Args:
        x (Tensor): Tensor to be smoothed, of size [*vol_shape, C]
        sigmas (scalar or list of scalars): gaussian sigma

    Returns:
        Tensor: a smoothed Tensor of size [*vol_shape, C]
    """

    # input checking
    if x.dtype != tf.float32:
        x = tf.cast(x, 'float32')
    shape = x.shape.as_list()
    ndim = len(shape) - 1
    if not isinstance(sigmas, (tuple, list)):
        sigmas = [sigmas] * ndim
    assert len(sigmas) == ndim, \
        'incorrect number of sigmas passed: %d (needed: %d)' % (len(sigmas), ndim)

    kernel = [ne.utils.gaussian_kernel(s) for s in sigmas]
    return ne.utils.separable_conv(x, kernel)


def batch_gaussian_smoothing(x, sigmas):
    """
    like gaussian_smoothing, but assume shape: [batch_size, *vol_shape, features]

    Essentially reshapes to [*vol_shape, ?], run gaussian_smoothing, and reshape back.

    Args:
        x (Tensor): Tensor of size [B, *vol_shape, C]
        sigmas (list or float): [description]

    Returns:
        Tensor: blurred Tensor of size [B, *vol_shape, C]
    """
    # input management
    ndim = len(x.shape) - 2
    assert ndim in range(1, 4), 'Dimension can only be in [1, 2, 3]'

    # reshape [B, *vol_shape, C] --> [*vol_shape, C * B]
    x_new = K.permute_dimensions(x, list(range(1, ndim + 2)) + [0])
    x_new = K.reshape(x_new, list(x.shape[1:ndim + 1]) + [-1])

    # nes.utils.gaussian_blur assumes vol_shape + [C] size, blurs along channel
    x_blur = gaussian_smoothing(x_new, sigmas)

    # reshape to [*vol_shape, C, B] to be able to move to keras dimensions below
    # to extract the batch size back, we use tensor shapes
    shape = tf.shape(x)
    new_shape = tf.concat([shape[1:], shape[0:1]], 0)
    field_smooth = K.reshape(x_blur, new_shape)

    # reshape back to [B, *vol_shape, C]
    return K.permute_dimensions(field_smooth, [ndim + 1] + list(range(ndim + 1)))


def plot_keras_hist_losses(hist, figsize=(15, 7), smart_zoom=-1, title=None):
    """
    plot losses from keras history object (returned when fitting a model)

    Example:
        hist = model.fit(...)
        plot_keras_hist_losses(hist)
        <pretty graphs appear>

    Parameters:
        fname: name of file to read history from (usually written with
               nes.callbacks.WriteHistory)
        figsize: figure size as passed to matplotlib.pyplot.figure.
            Default is (15,7), a nice size for jupyter windows.
        smart_zoom: if < 0, then will not zoom in. if >= 0 and < 1,
            will show that percentage of epochs.
        if >= 1, has to be integer, will show that many last epochs
        title: if a string will be used as the figure title (must be unique!)
               if a list of strings will be used as the subplot title
               for each subplot

    Author: brf
    """

    # check that we have material to plot
    assert len(hist.epoch) > 0, 'could not find any epochs to plot'

    # prepare window
    if type(title) is str:
        plt.figure(figsize=figsize, num=title)
    else:
        plt.figure(figsize=figsize)

    total_subplots = 2 + 1 * (smart_zoom >= 0)

    # plot losses with some labeling
    plt.subplot(1, total_subplots, 1)
    for key in hist.history.keys():
        plt.plot(hist.epoch, hist.history[key])
    plt.legend(hist.history.keys())
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    if type(title) is list:
        plt.title(title[0])

    # plot only loss and validation loss, with some labeling
    plt.subplot(1, total_subplots, 2)
    plt.plot(hist.epoch, hist.history['loss'], '.-k')
    legend_labels = ['loss']
    if 'val_loss' in hist.history:
        plt.plot(hist.epoch, hist.history['val_loss'], '.-b')
        legend_labels.append('val_loss')
    plt.legend(legend_labels)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    if type(title) is list and len(title) > 1:
        plt.title(title[1])

    # "smart" epoch-zoom: only plot the last few epochs
    # this is all very experimental
    if smart_zoom >= 0:
        if smart_zoom < 1:
            start = len(hist.epoch) - np.ceil(len(hist.epoch) * smart_zoom).astype(int)
        else:
            start = int(smart_zoom)

        plt.subplot(1, total_subplots, 3)

        plt.plot(hist.epoch[start:], hist.history['loss'][start:], '.-k')
        legend_labels = ['loss']
        if 'val_loss' in hist.history:
            plt.plot(hist.epoch[start:], hist.history['val_loss'][start:], '.-b')
            legend_labels.append('val_loss')
        plt.legend(legend_labels)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.grid()
        if type(title) is list and len(title) > 2:
            plt.title(title[2])

    plt.show(block=False)


def box_smooth(y, filter_width):
    """
    apply a box/moving average filter to a 1D numpy array
    parameters:
        y:            the 1D numpy array to smooth
        filter_width: the size of the box filter (must be odd)
    """

    if filter_width % 2 == 0:
        warnings.warn('box_smooth called with even filter width %d' % filter_width)

        filter_width += 1
    box = np.ones(filter_width) / filter_width
    whalf = (filter_width - 1) // 2
    y_smooth = np.convolve(np.pad(y, (whalf, whalf), mode='reflect'), box, mode='valid')
    return y_smooth


def plot_fit_callback(fnames,
                      figsize=(15, 7),
                      title=None,
                      keys=None,
                      smart_zoom=-1,
                      plot_block=False,
                      close_all=False,
                      return_data=False,
                      legend=None,
                      xlim=None,
                      ylim=None,
                      linewidth=2,
                      xlabel=None,
                      ylabel=None,
                      title_font_size=12,
                      label_font_size=14,
                      skip_epochs=0,
                      remove_outlier_thresh=None,
                      outlier_whalf=2,
                      outlier_replace_val=None,
                      smooth=None):
    """
    example:
        plot_fit_callback('history.txt', keys=['loss', 'lr'])

    parameters:utils.py

        keys:  a list of keywords to plot. Should be a subset of the ones
               that are in the log dictionary (ones not in the keys list will
               not be plotted)
        title: if a string will be used as the figure title (must be unique!)
               if a list of strings will be used as the subplot title
               for each subplot
        smart_zoom: if < 0, then will not zoom in. if >= 0 and < 1,
            will show that percentage of epochs.
            if >= 1, has to be integer, will show that many last epochs
        figsize: figure size as passed to matplotlib.pyplot.figure.
            Default is (15,7), a nice size for jupyter windows.
        plot_block: passed to matlabplotlib.pyplot.show (default=False)
        skip_epochs: number of initial epochs to skip when plotting (default=0)
        smooth: if not None, specify an odd int to apply a boxcar filter
        remove_outlier_thresh:  if not None set values more than remove_outlier_thresh
        outlier_whalf -  avg of nbrs to take median over
        outlier_replace_val - replace outliers with this value in the plot instead of median

    author: brf2
    """

    # read the file and separate it into columns
    if type(fnames) is not list:
        fnames = [fnames]

    if close_all:
        plt.close('all')

    axes = []
    return_vals = []

    for fno, fname in enumerate(fnames):
        fp = open(fname, 'r')
        rows = fp.read().split('\n')
        fp.close()
        headers = rows[0]
        names = headers[2:].split(' ')
        rows = rows[1:-1]
        ncols = len(rows[0].split(' ')[:-1])
        cols = [[] for col in range(ncols)]

        for row in rows:
            for cno, val in enumerate(row.split(' ')[:-1]):
                cols[cno].append(float(val))

        if title is None:
            title = fname

        if fno == 0:
            plt.figure(figsize=figsize, num=title)

        # plot each column in the file in a different subplot row

        # if keys is provided as a parameter go through the list of all
        # available ones and mark those provided by the caller as
        # not skipped
        if keys is None:
            skipped = np.zeros(ncols)   # use all keys
        else:
            skipped = np.ones(ncols)
            for key in keys:
                reg = re.compile(key)
                ind = [(reg.match(name) is not None) for name in names].index(True)
                if ind >= 0:
                    skipped[ind] = 0

        # take the epochs out of the columns
        epochs = cols[0]
        cols = cols[1:]
        names = names[1:]
        skipped = skipped[1:]
        nplots = (skipped == 0).sum()
        pno = 0
        if smart_zoom >= 0:
            if smart_zoom < 1:
                start = len(epochs) - np.ceil(len(epochs) * smart_zoom).astype(int)
            else:
                start = int(smart_zoom)
        else:
            start = skip_epochs

        # plot each remaining column in a different subplot
        for cno, col in enumerate(cols):
            if skipped[cno]:
                continue
            if remove_outlier_thresh is not None:
                for ind in range(start, len(col) - 1):
                    mlist = []
                    for ind2 in range(max(0, ind - outlier_whalf),
                                      min(ind + outlier_whalf + 1, len(col))):
                        mlist.append(col[ind2])
                    mval = np.nanmedian(mlist)

                    if abs(col[ind]) > abs(remove_outlier_thresh * mval) or np.isnan(col[ind]):
                        col[ind] = mval if outlier_replace_val is None else outlier_replace_val

            pno += 1
            if smooth is not None:
                data = box_smooth(col[start:], smooth)
            else:
                data = col[start:]

            if return_data:
                return_vals.append(data)

            if fno == 0:
                axes.append(plt.subplot(nplots, 1, pno))
                plt.xlabel('epoch', fontsize=label_font_size)
                if ylabel is None:
                    plt.ylabel(names[cno], fontsize=label_font_size)
                else:
                    if isinstance(ylabel, list):
                        yl = ylabel[pno-1]
                    else:
                        yl = ylabel
                    plt.ylabel(yl, fontsize=label_font_size)

                if title is not None:
                    if isinstance(title, list):
                        plt.title(title[pno-1])
                    elif pno == 1:  # only on the first plot
                        plt.title(title)

                plt.grid()
            else:
                axes.append(plt.subplot(nplots, 1, pno))
                # plt.subplot(axes[pno - 1], label=fnames[fno])

            if xlim is not None:
                if type(xlim[0]) is list:
                    plt.xlim(xlim[pno - 1])
                else:
                    plt.xlim(xlim)
            if ylim is not None:
                if type(ylim[0]) is list:
                    plt.ylim(ylim[pno - 1])
                else:
                    plt.ylim(ylim)

            plt.plot(epochs[start:], data, linewidth=linewidth)
            if legend is not None:
                plt.legend(legend)

        plt.rcParams.update({'font.size': title_font_size})
        #if ylabel is not None:
        #    plt.ylabel(ylabel, fontsize=label_font_size)

        #if xlabel is not None:
        #    plt.xlabel(xlabel, fontsize=label_font_size)

    plt.show(block=plot_block)
    if return_data:
        return return_vals


@tf.function
def sphere_vol(vol_shape, radius, center=None, dtype=tf.bool):
    """
    draw nd sphere volume, similar to pystrum.pynd.ndutils.sphere_vol
    TODO: note, this is not faster than the pynd for small images

    Args:
        vol_shape (list): volume shape, a list of integers
        center (list or int): list or integer, if list then same length as vol_shape list
        radius (float): radius of the circle
        dtype (np.dtype): np.bool (binary sphere) or np.float32 (sphere with partial volume at edge)

    Returns:
        [np.bool or np.float32]: bw sphere, either 0/1 (if bool) or [0,1] if float32
    """

    # prepare inputs
    assert isinstance(vol_shape, (list, tuple)), 'vol_shape needs to be a list or tuple'
    ndims = len(vol_shape)

    if not isinstance(center, (list, tuple)):
        if center is None:
            center = [(f - 1) / 2 for f in vol_shape]
        else:
            center = [center] * ndims
    else:
        assert len(center) == ndims, "center list length does not match vol_shape length"

    # check dtype
    assert dtype in [tf.bool, tf.float32], 'dtype should be np.bool, np.float32'

    # prepare mesh
    mesh = ne.utils.volshape_to_ndgrid(vol_shape)
    centered_mesh = [(tf.cast(mesh[f], tf.float32) - center[f])**2 for f in range(ndims)]
    dist_from_center = K.sqrt(K.sum(tf.stack(centered_mesh, ndims), ndims))

    # create sphere
    sphere = dist_from_center <= radius
    if dtype == tf.float32:  # enable partial volume at edge
        float_sphere = tf.cast(sphere, tf.float32)
        df = radius - dist_from_center
        edge = tf.cast(tf.logical_and(df < 0, df > -1), tf.float32)
        sphere = float_sphere + edge * (1 + df)

    # done!
    return sphere


@tf.function
def random_smooth_deformation_field(vol_shape,
                                    amplitude,
                                    noise_method='normal',
                                    smooth_method='blur',
                                    small_vol_shape=None,
                                    sigmas=None):
    """

    Generate random smooth deformation fields.

    # TODO: in smooth_method = 'blur', correct std analyticially not empirically.

    See also batch_gaussian_smoothing

    Args:
        vol_shape ([type]): [description]
        amplitude ([type]): [description]
        noise_method (str, optional): [description]. Defaults to 'normal'.
        smooth_method (str, optional): [description]. Defaults to 'blur'.
        small_vol_shape ([type], optional): [description]. Defaults to None.
        sigmas ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    ndims = len(vol_shape)
    vol_shape = vol_shape + [ndims]

    assert smooth_method in ['blur', 'zoom', 'resize'], \
        'smooth_method can only be blur, zoom, or resize'
    assert noise_method in ['normal', 'uniform'], 'noise_method can only be normal or uniform'

    # prepare random volume shape
    if smooth_method == 'blur':
        rand_shape = vol_shape
    else:
        assert small_vol_shape is not None, \
            'small_vol_shape cannot be None if smooth_method is zoom/resize'
        small_vol_shape = small_vol_shape + [ndims]
        rand_shape = small_vol_shape

    if noise_method == 'normal':
        vol = tf.random.normal(rand_shape, dtype=tf.float32)
    else:
        vol = tf.random.uniform(rand_shape, dtype=tf.float32)

    if smooth_method == 'blur':
        assert sigmas is not None, 'sigmas cannot be be none if smooth_method is blur'
        blurred_vol = gaussian_smoothing(vol, sigmas)

        # need to correct the std since the blur will destroy it
        # currently doing this empirically, but there is probably a closed form solution based on
        # some signal processing magic.
        blurred_vol = blurred_vol * amplitude / tf.math.reduce_std(blurred_vol)

        # update: corrective factor is: std(rms(ndim_gaussian_kernel))
        # but this actually seems a bit slower, and does not work perfectly due to edge effects
        # would need to change implementation of edge
        #   if not isinstance(sigmas, (list, tuple)):
        #       sigmas = [sigmas] * ndims
        #   A = K.flatten(ne.utils.gaussian_kernel(sigmas))
        #   blurred_vol = blurred_vol * amplitude / K.sqrt(K.sum(K.square(A)))

    else:  # resizing
        zoom_factor = [vol_shape[d] / small_vol_shape[d] for d in range(len(vol_shape) - 1)]
        blurred_vol = ne.utils.zoom(vol * amplitude, zoom_factor)

    return blurred_vol


def patches_to_image_2D(patches, image_size, strides=(8, 8)):
    """
    patches should be shape (width, height, nchannels, npatches) (see extract_patches)

    Recover image from tf.image.extract_patches.
    patches - tf.constant with shape - [n_images, height, width, 1]
    image_size - tuple (height,width) original image size (before slicing)
    n_images - tuple (horizontal, vertical) number of patches

    Author: brf2
    """
    patch_size = patches.shape[0:2]
    n_images = [int((image_size[i] - patch_size[i]) / strides[i] + 1) for i in range(len(strides))]
    # Kernel for counting overlaps
    kernel_ones = tf.ones(patch_size, dtype=tf.int8)

    # Initialize image
    recovered = tf.zeros(image_size)
    counts = tf.zeros(image_size, dtype=tf.int8)
    chlist = []
    mlist = []
    for ch in range(image_size[-1]):
        recovered_ch = tf.zeros(image_size[0:-1])
        counts_ch = tf.zeros(image_size[0:-1], dtype=tf.int8)
        frame = 0
        for j in range(n_images[0]):
            for i in range(n_images[1]):
                # Make indices from meshgrid
                row_start = strides[0] * j
                row_end = patch_size[0] + strides[0] * j
                col_start = strides[1] * i
                col_end = patch_size[1] + strides[1] * i
                indices = tf.meshgrid(tf.range(row_start, row_end),
                                      tf.range(col_start, col_end), indexing='ij')
                indices = tf.stack(indices, axis=-1)
                # Add sliced image to recovered image indice
                recovered_ch = tf.tensor_scatter_nd_add(recovered_ch,
                                                        indices,
                                                        patches[..., ch, frame])
                # Update counts
                counts_ch = tf.tensor_scatter_nd_add(counts_ch, indices, kernel_ones)
                frame += 1

        mlist.append(counts_ch)
        chlist.append(recovered_ch)

    recovered = tf.stack(chlist, axis=-1)
    counts = tf.stack(mlist, axis=-1)
    recovered = tf.math.divide_no_nan(recovered, tf.cast(counts, tf.float32))

    return recovered


def extract_patches_from_image_2D(image, psize=32, stride=8, rate=1, padding='VALID', name=None):
    """
    images should be shape (width, height, nchannels)

    Recover image from tf.image.extract_patches.
    patches  - tf.constant with shape - [n_images, height, width, 1]
    psize    - size of each patch
    stride   - stride between patches
    padding  - see tf.image.extract_patches
    name     - see tf.image.extract_patches
    returns patches as (psize, psize, nchannels, npatches)

    Author: brf2
    """

    # process each output channel separately
    image = tf.squeeze(image)  # in case caller included a singleton batch dim
    chlist = []
    for ch in range(image.shape[-1]):
        slices = tf.image.extract_patches(
            image[tf.newaxis, ..., ch:ch + 1],
            (1, psize, psize, 1),
            (1, stride, stride, 1),
            (1, rate, rate, 1),
            padding=padding,
            name=name
        )
        patches = tf.reshape(slices, (-1, psize, psize))
        chlist.append(patches)

    # put the channels dimension at the end
    patches = tf.transpose(tf.stack(chlist, axis=0), (2, 3, 0, 1))

    return patches


def check_and_compile(
        model,
        gen=None,
        check_losses=False,
        check_layers=True,
        extra_outputs=None,
        **kwargs):
    """
    function to check shapes of network inputs/outputs and
    generator returns to make sure that they match. Also checks
    losses and loss weights, runs the generator once and checks the
    inputs and outputs to make sure that they are finite, then runs
    model.predict(inputs) and checks the prediction for NaNs/infs also
    (if check_losses==True)

    parameters:
       model         - input model to check and compile
       gen           - generator to use to check shape of inputs and outputs
       check_losses  - boolean (if true checkes len of losses)
       check_layers  - boolean (if true checks that all layer outputs are keras tensors)
       extra_outputs - list of ints to subtract from model outputs. Useful if you are
                       stacking something into the output to be used in the loss like a mask

    Author: brf2
    """

    fname = 'check_and_compile'
    key_list = list(kwargs.keys())
    val_list = list(kwargs.values())
    if 'loss' in kwargs.keys():
        loss_index = key_list.index('loss')
        losses = val_list[loss_index]
        assert len(losses) == len(model.outputs), \
            '%s: len(losses)=%d does not match # of model outputs %d' \
            % (fname, len(losses), len(model.outputs))

    if check_layers:
        for lno, layer in enumerate(model.layers):
            outputs = layer.get_output_at(-1)
            if type(outputs) is not list:
                outputs = [outputs]
            for toutput in outputs:
                assert tf.keras.backend.is_keras_tensor(toutput), \
                    f'layer {lno}: {layer.name} output is not a symbolic tensor'
            inputs = layer.get_input_at(-1)
            if type(inputs) is not list:
                inputs = [inputs]

            for tinput in inputs:
                assert tf.keras.backend.is_keras_tensor(tinput), \
                    f'layer {lno}: {layer.name} input is not a symbolic tensor'

    if 'loss_weights' in kwargs.keys():
        loss_wt_index = key_list.index('loss_weights')
        loss_weights = val_list[loss_wt_index]
        assert len(loss_weights) == len(model.outputs), \
            '%s: len(loss_weights)=%d  does not match # of model outputs %d' \
            % (fname, len(loss_weights), len(model.outputs))

    if gen is not None:  # match gen inputs/outputs with model
        inb, outb = next(gen)
        if type(inb) is not list:
            inb = [inb]
        if type(outb) is not list:
            outb = [outb]
        assert len(inb) == len(model.inputs), \
            '%s: len(inb)=%d != len(model.inputs) %d' % \
            (fname, len(inb), len(model.inputs))
        assert len(outb) == len(model.outputs), \
            '%s: len(outb)=%d != len(model.outputs) %d' % \
            (fname, len(outb), len(model.outputs))
        for ino in range(len(inb)):
            in_shape = tuple(model.inputs[ino].get_shape().as_list()[1:])
            assert in_shape == np.array(inb[ino]).shape[1:], \
                '%s: input %d shape does not match model input shape:' % \
                (fname, ino) + \
                'inb = ' + str(inb[ino].shape) + ' model = ' + str(in_shape)

        if extra_outputs is None:
            extra_outputs = [0] * len(outb)

        for ono in range(len(outb)):
            out_shape = model.outputs[ono].get_shape().as_list()[1:]
            out_shape[-1] -= extra_outputs[ono]
            out_shape = tuple(out_shape)

            assert out_shape == np.array(outb[ono]).shape[1:], \
                '%s: output %d shape does not match model output shape:' % \
                (fname, ono) + 'outb = ' + \
                str(outb[ono].shape) + ' model = ' + str(out_shape)

        if check_losses and ('loss' in kwargs.keys()):
            loss_index = key_list.index('loss')
            losses = val_list[loss_index]
            pred = model.predict(inb)
            if type(pred) is not list:
                pred = [pred]
                outb = [outb]
                inb = [inb]
            for pno, pred_out in enumerate(pred):
                # y_true = tf.convert_to_tensor(outb[pno], dtype=tf.float32)
                # y_pred = tf.convert_to_tensor(pred_out, dtype=tf.float32)
                # lval = losses[pno](y_true, y_pred) # unused
                assert np.isfinite(pred_out).min() is False, \
                    'predicted output %d contains non-finite values' % pno
                assert np.isfinite(inb[pno]).min() is False, \
                    'generated input %d contains non-finite values' % pno
                assert np.isfinite(outb[pno]).min() is False, \
                    'generated output %d contains non-finite values' % pno
                assert np.isfinite(outb[pno]).min() is False, \
                    'loss %d contains non-finite values - ' % pno + \
                    str(losses[pno])

    return model.compile(**kwargs)


def draw_diffeomorphism(vol_shape, max_std, modulate=False, int_steps=5):
    """
    Synthesize a deformation field by integrating a random stationary velocity
    field (SVF). The SVF will be sampled and integrated at half resolution,
    then upsampled. Random blurring will be applied to the deformation.

    The generated warps look more natural/bubbly than those created for
    SynthMorph by drawing at different scales, upsampling and adding. However,
    they do not appear to perform as well as the SynthMorph strategy for
    learning registration from random shapes using SVF scales 1/8, 1/16 and
    1/32. Blurring the warps worked better than the SVF, which resulted in very
    weak deformations while increasing the SD resulted in a proportion of warps
    that were too extreme.

    Parameters:
        vol_shape: Input image shape without batch or feature dimension.
        max_std: Standard deviation (SD) used for sampling the SVF.
        modulate: Whether to sample the SD uniformely from (0, max].
        int_steps: Number of integration steps.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    import voxelmorph as vxm

    num_dim = len(vol_shape)
    vol_shape = np.asarray(vol_shape)
    vel_shape = (*vol_shape // 2, num_dim)

    # Deformation.
    max_std = np.asarray(max_std, dtype='float32')
    vel_std = tf.random.uniform((1,), maxval=max_std) if modulate else max_std
    vel_field = tf.random.normal(vel_shape, stddev=vel_std)
    def_field = vxm.utils.integrate_vec(vel_field, nb_steps=int_steps)

    # Constant grid.
    max_blur = 5
    width = np.round(max_blur * 3) * 2 + 1
    cen = (width - 1) / 2
    mesh = np.arange(width) - cen
    mesh = -0.5 * mesh**2
    mesh = [tf.constant(mesh, dtype='float32') for _ in range(num_dim)]

    # Blurring kernels.
    sigma = [tf.random.uniform((1,), minval=2, maxval=max_blur)] * num_dim
    exponent = [m / s**2 for m, s in zip(mesh, sigma)]
    kernels = [tf.exp(x) for x in exponent]
    kernels = [x / tf.reduce_sum(x) for x in kernels]

    # Blur and resize.
    def_field = ne.utils.separable_conv(def_field, kernels)
    def_field = ne.utils.resize(def_field * 2, zoom_factor=2)

    return def_field


def mask_encoding(x, depth):
    """ encode an integer in a mask Tensor

    e.g. x=3 with depth=5 --> Tensor bool([1, 1, 1, 0, 0])

    Args:
        x ([type]): [description]
        depth ([type]): [description]
    """

    if not (isinstance(x, tf.Tensor) and x.dtype == 'int32'):
        x = tf.cast(x, tf.int32)

    shape = tf.concat([tf.shape(x), tf.ones(1, tf.int32) * depth], 0)

    # need to expand
    flatx = K.flatten(x)

    def me(xi):
        true = tf.ones(xi, dtype=tf.bool)
        false = tf.zeros(depth - xi, dtype=tf.bool)
        return tf.concat([true, false], 0)

    mask = tf.map_fn(me, flatx, dtype=tf.bool)
    return tf.reshape(mask, shape)


def downsample_axis_interpn(x, ds, axis=0, start=None, return_mask=False, interp_method='linear'):
    """ downsample along axis and interpolate missing values

    # TODO: add blur along dimension first

    Example:
        # make a blob
        blob = nes.utils.data.blob_stack([64, 64], 16, 5.5, 4, 1)[...,0]

        # downsample
        blob_ds, mask = downsample_axis_interpn(blob, 10, start=3, axis=0, return_mask=True)

        # visualize
        slices = [f.numpy() for f in [blob, blob_ds, mask]]
        ne.plot.slices(slices, do_colorbars=True, cmaps=['gray']);

        # make sure images are equal clean within mask
        np.all(np.isclose(blob[mask.numpy()].numpy(), blob_ds[mask.numpy()].numpy()))

    Args:
        x (tf.Tensor): nD Tensor to interpolate. size vol_shape.
        ds (int): downsample amount
        axis (int, optional): [description]. Defaults to 0.
        start (int, optional): first True index
        return_mask (bool, optional): [description]. Defaults to False.
        interp_method (str, optional): [description]. Defaults to 'linear'.

    Returns:
        [type]: [description]
    """

    # input parsing
    assert axis < len(x.shape), 'incorrect axis'
    ds_tf = tf.cast(ds, tf.float32)

    if start is None:
        start = np.random.randint(0, ds)

    start_tf = tf.cast(start, tf.float32)

    # extract data to keep
    ind = range(start, x.shape[axis], ds)
    x_data = ne.utils.take(x, ind, axis)

    # interpolate missing data
    loc = [tf.cast(f, tf.float32) for f in ne.utils.volshape_to_ndgrid(x.shape)]
    loc[axis] = loc[axis] / ds_tf - start_tf / ds_tf
    interp_vol = ne.utils.interpn(x_data, loc, interp_method=interp_method)

    # prepare outputs and mask.
    # could do mask with tf.Variable but this turned out to be harder than it looked when using
    #  tf.function decorator
    outputs = [interp_vol]
    if return_mask:
        mask = np.zeros(x.shape)
        ind = np.arange(start, x.shape[axis], ds)
        resh = [1] * len(x.shape)
        resh[axis] = ind.shape[0]
        ind = np.reshape(ind, resh)
        np.put_along_axis(mask, ind, 1, axis)
        outputs.append(tf.convert_to_tensor(mask, tf.bool))

    return outputs


def print_if_true(cond, print_tensor, **kwargs):
    """
    print a tensor if given condition is true

    Author @avd12 for @brf2
    """
    printfn = lambda: tf.print(print_tensor, **kwargs)
    empty_print_fn = tf.print(None, output_stream='file:///dev/null')
    return tf.cond(cond, true_fn=printfn, false_fn=lambda: empty_print_fn)


def find_layers_with_substring(model, substr):
    """
    find all layers that contain a substring and return a list of them
    """

    target_layers = []
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            target_layers += find_layers_with_substring(layer, substr)
        else:
            if substr in layer.name:
                target_layers.append(layer)

    return target_layers


def find_layers_unique_to_model(model_in, model_not_in, val=0):
    '''
    find layers that are in one model but not another. This is useful
    when using input_models to a second model for finding layers that are
    in the second model but not the input model

    val is an optional value to put into the dictionary such as a learning rate
    '''
    ldict = {}
    for layer in model_in.layers:
        if model_not_in is None or layer not in model_not_in.layers:
            ldict[layer.name] = val
            if hasattr(layer, 'layers'):
                dict2 = find_layers_unique_to_model(layer, model_not_in, val)
                ldict.update(dict2)

    return ldict


def scale_layer_weights(model, scale):
    """
    scale all the weights in a model by the specified amount
    """

    for layer in model.layers:
        w = layer.get_weights()
        w = [wi * scale for wi in w]
        layer.set_weights(w)
        if hasattr(layer, 'layers'):
            scale_layer_weights(layer, scale)

    return


def morphology_3d(label_map, label, niter=1, operation='dtrans', eight_connectivity=True,
                  rand_crop=None):
    '''
    given a label map of indices with shape (width,height,depth,channels)
    peform an operation on the voxels with index == label niter times.
    Supported operations are:
    'dtrans' - return the narrow band distance transform from label
               (<0 in the interior, >0 in the exterior, all voxels
                more than niter from the boundary set to niter+1)
    'erode' - erode the label
    'dilate'- dilate the label
    rand_crop - in [0,1] the max fraction of 1s in the filter to randomly set to 0

    Note: batch_dim should not be included in label_map, but channel dim should be
    '''

    # conv ops expect a batch dim so add it in and binarize image
    if operation == 'open':
        lm1 = morphology_3d(label_map, label, niter=niter, operation='erode',
                            eight_connectivity=eight_connectivity,
                            rand_crop=rand_crop)
        lm2 = morphology_3d(tf.cast(lm1, label_map.dtype), label, niter=niter,
                            operation='dilate',
                            eight_connectivity=eight_connectivity,
                            rand_crop=rand_crop)
        return lm2
    elif operation == 'close':
        lm1 = morphology_3d(label_map, label, niter=niter, operation='dilate',
                            eight_connectivity=eight_connectivity,
                            rand_crop=rand_crop)
        lm2 = morphology_3d(tf.cast(lm1, label_map.dtype), label, niter=niter,
                            operation='erode', eight_connectivity=eight_connectivity,
                            rand_crop=rand_crop)
        return lm2

    label_map = tf.equal(label_map[tf.newaxis, ...], label)
    ndims = len(label_map.shape) - 2

    conv_fn = getattr(tf.nn, 'conv%dd' % ndims)
    prev_outside = tf.cast(label_map, dtype=tf.float32)
    prev_inside = tf.cast(label_map, dtype=tf.float32)
    strides = [1] * (ndims + 2)
    dtrans = tf.zeros(tf.shape(label_map), dtype=tf.float32)

    # allocate a box-car conv filter for dilating fg and bg
    if eight_connectivity:
        sum_filt = np.zeros(ndims * [3] + [1, 1])
        if ndims == 3:
            sum_filt[1, 1, 1, 0, 0] = 1
            sum_filt[1, 1, 2, 0, 0] = 1
            sum_filt[2, 1, 1, 0, 0] = 1
            sum_filt[0, 1, 1, 0, 0] = 1
            sum_filt[1, 1, 0, 0, 0] = 1
            sum_filt[1, 2, 1, 0, 0] = 1
            sum_filt[1, 0, 1, 0, 0] = 1
        else:    # should be 4-connected in 2d
            sum_filt[0, 1, 0, 0] = 1
            sum_filt[1, 0, 0, 0] = 1
            sum_filt[1, 2, 0, 0] = 1
            sum_filt[2, 1, 0, 0] = 1

        sum_filt = tf.convert_to_tensor(sum_filt, dtype=tf.float32)
    else:
        sum_filt = tf.ones(ndims * [3] + [1, 1])

    sum_filt_orig = tf.identity(sum_filt)

    tf_false = False
    inverse_image = tf.cast(tf.equal(label_map, tf_false), tf.float32)
    if rand_crop is not None:
        if isinstance(rand_crop, list):
            min_crop = rand_crop[0]
            max_crop = rand_crop[1]
        else:
            max_crop = rand_crop
            min_crop = 0

        rand = tf.random.get_global_generator()
        inds = tf.random.shuffle(tf.where(sum_filt_orig))
        max_zero = tf.cast(tf.cast(len(inds), tf.float32) *
                           tf.cast(max_crop, tf.float32), tf.int32)
        min_zero = tf.cast(tf.cast(len(inds), tf.float32) *
                           tf.cast(min_crop, tf.float32), tf.int32)
        nzero = tf.reshape(rand.uniform((1,), min_zero, max_zero, dtype=tf.int32), [])
        inds = inds[0:nzero]
        updates = tf.gather_nd(tf.ones(sum_filt_orig.shape), inds)
        snd = tf.scatter_nd(inds, updates, sum_filt_orig.shape)
        sum_filt = sum_filt_orig - snd    # zero out those indices
        center_on = np.zeros(sum_filt.shape)
        center_on[1, 1, 1, 0, 0] = 1
        sum_filt = sum_filt + tf.convert_to_tensor(center_on, dtype=sum_filt.dtype)
        sum_filt = tf.clip_by_value(sum_filt, 0, 1)

    for bval in range(1, niter + 1):
        # randomly remove some of the 1s in the dilation filter to give a random appearance
        dilated = conv_fn(prev_outside, sum_filt, strides, 'SAME')
        outside = tf.cast(tf.greater(dilated, 0), tf.float32)
        if bval == 1 and rand_crop is not None and False:

            import freesurfer as fs
            import pdb as gdb

            fv = fs.Freeview()
            fv.vol(tf.squeeze(prev_outside), name='prev_outside', opts=':visible=0')
            fv.vol(tf.squeeze(sum_filt), name='sum_filt', opts=':visible=0')
            fv.vol(tf.squeeze(dilated), name='dilated', opts=':visible=0')
            fv.vol(tf.squeeze(outside), name='outside', opts=':visible=1')
            fv.show(title=f'iter{bval}')
            gdb.set_trace()

        inside = tf.cast(tf.equal(conv_fn(inverse_image, sum_filt, strides, 'SAME'), 0), tf.float32)
        # compute the next inside and outside rings
        inside_border = prev_inside - tf.cast(inside, tf.float32)
        outside_border = tf.cast(outside, tf.float32) - prev_outside
        prev_outside = tf.cast(outside, tf.float32)
        prev_inside = tf.cast(inside, tf.float32)
        inverse_image = tf.cast(tf.equal(prev_inside, 0), dtype=tf.float32)
        dtrans = tf.add(dtrans, tf.cast(bval, tf.float32) * outside_border)
        dtrans = tf.add(dtrans, tf.cast(-bval, tf.float32) * inside_border)

    # set everything more than border voxels from the boundary to +/- border+1
    interior = tf.cast(tf.math.logical_and(tf.equal(dtrans, 0), label_map), tf.float32)
    exterior = tf.cast(tf.math.logical_and(tf.equal(dtrans, 0),
                                           tf.equal(label_map, tf_false)), tf.float32)
    dtrans = tf.add(dtrans, -(tf.cast(niter, tf.float32) + 1) * interior)
    dtrans = tf.add(dtrans, (tf.cast(niter, tf.float32) + 1) * exterior)

    if operation == 'dtrans':
        return_image = dtrans
    elif operation == 'dilate':
        return_image = dtrans < (tf.cast(niter, dtype=dtrans.dtype) + .5)
    elif operation == 'erode':
        return_image = dtrans < -(tf.cast(niter, dtype=dtrans.dtype) + .5)
    else:
        assert 0, f'morph3D: unknown operation {operation}'

    return return_image[0, ...]  # remove dimension we added


def dilate_likely_voxels(label_vol, intensity_vol, label, ndil=1, mdist_ratio=1, whalf=3,
                         eight_connectivity=True):
    '''
    dilate a label volume into adjacent voxels that are closer in intensity to the
    nearby labeled voxels than to nearby unlabeled voxels
    labels are changed if mdist_ratio * abs(label_mean-i0)/label_std is less than
    abs(non_label_mean-i0)/non_label_std, were the means and stds are computed in
    a 2*whalf+1 sized window around each voxel
    '''

    inshape = label_vol.shape
    for dil in range(ndil):
        tvol = tf.convert_to_tensor(label_vol[..., np.newaxis])
        dil_vol = morphology_3d(tvol, label, 1, operation='dilate',
                                eight_connectivity=eight_connectivity)
        dil_vol = dil_vol.numpy().squeeze()
        out_vol = label_vol.copy()
        bvol = np.logical_and(dil_vol, np.logical_not(label_vol))
        inds = zip(*np.where(bvol))
        nadded = 0
        for x, y, z in inds:
            # xk, yk, zk = np.mgrid[x-whalf:x+whalf+1, y-whalf:y+whalf+1, z-whalf:z+whalf+1]
            vlist_on = []
            vlist_off = []
            for xk in range(max(x - whalf, 0), min(x + whalf + 1, inshape[0])):
                for yk in range(max(y - whalf, 0), min(y + whalf + 1, inshape[1])):
                    for zk in range(max(z - whalf, 0), min(z + whalf + 1, inshape[2])):
                        if label_vol[xk, yk, zk] == label:
                            vlist_on.append(intensity_vol[xk, yk, zk])
                        else:
                            vlist_off.append(intensity_vol[xk, yk, zk])
            if len(vlist_on) == 0 or len(vlist_off) == 0:
                continue  # should never happen

            mean_on = np.array(vlist_on).mean()
            mean_off = np.array(vlist_off).mean()
            if len(vlist_on) < 3 or len(vlist_off) < 3:  # don't use std
                on_dist = abs(mean_on - intensity_vol[x, y, z])
                off_dist = abs(mean_off - intensity_vol[x, y, z])
            else:
                std_on = max(.1, np.array(vlist_on).std())
                std_off = max(.1, np.array(vlist_off).std())
                on_dist = abs(mean_on - intensity_vol[x, y, z]) / std_on
                off_dist = abs(mean_off - intensity_vol[x, y, z]) / std_off

            if on_dist * mdist_ratio < off_dist:
                nadded += 1
                out_vol[x, y, z] = label

        print(f'{nadded} labels added in dilation {dil+1} of {ndil}')
        if dil < ndil - 1:
            label_vol = out_vol.copy()

    return out_vol


def set_trainable(model, trainable):
    '''
    set all model layers.trainable to the specified value. This will properly follow
    embedded models recursively if needed
    '''
    model.trainable = trainable
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            set_trainable(layer, trainable)
        layer.trainable = trainable


# Based on np.linalg.cond(x, p=None)
def tf_cond(x):
    x = tf.convert_to_tensor(x)
    s = tf.linalg.svd(x, compute_uv=False)
    r = s[..., 0] / s[..., -1]
    # Replace NaNs in r with infinite unless there were NaNs before
    x_nan = tf.reduce_any(tf.math.is_nan(x), axis=(-2, -1))
    r_nan = tf.math.is_nan(r)
    r_inf = tf.fill(tf.shape(r), tf.constant(math.inf, r.dtype))
    tf.where(x_nan, r, tf.where(r_nan, r_inf, r))
    return r


def find_deepest_layer(unet):
    ndims = len(unet.outputs[0].shape) - 2
    Conv = getattr(KL, 'Conv%dD' % ndims)
    min_elts = 1e12
    min_layer = None
    for layer in unet.layers:
        if type(layer) == Conv:
            nelts = np.prod(layer.output.shape.as_list()[1:])
            if nelts <= min_elts:
                min_elts = nelts
                min_layer = layer

    return min_layer


def is_invertible(x, epsilon=1e-6):  # Epsilon may be smaller with tf.float64
    x = tf.convert_to_tensor(x)
    eps_inv = tf.cast(1 / epsilon, x.dtype)
    x_cond = tf_cond(x)
    return tf.math.is_finite(x_cond) & (x_cond < eps_inv)


def is_not_invertible(x, epsilon=1e-6):  # Epsilon may be smaller with tf.float64
    x = tf.convert_to_tensor(x)
    eps_inv = tf.cast(1 / epsilon, x.dtype)
    x_cond = tf_cond(x)
    return tf.math.is_nan(x_cond) | (x_cond >= eps_inv)


def subsample_axis(x, stride_min=1, stride_max=1, axes=None, prob=1, upsample=True, seed=None):
    '''
    Symmetrically subsample a tensor by a factor f (stride) along a single axis
    using nearest-neighbor interpolation and optionally upsample again, to reduce
    its resolution. Both f and the subsampling axis can be randomly drawn.

    Arguments:
        x: Input tensor or NumPy array of any type.
        stride_min: Lower bound on the subsampling factor.
        stride_max: Upper bound on the subsampling factor.
        axes: Tensor axes to draw the subsampling axis from. None means all axes.
        prob: Subsampling probability. A value of 1 means always, 0 never.
        upsample: Upsample the tensor to restore its original shape.
        seed: Integer for reproducible randomization.

    Returns:
        Tensor with randomly thick slices along a random axis.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    '''
    # Validate inputs.
    if not tf.is_tensor(x):
        x = tf.constant(x)
    rand = np.random.default_rng(seed)
    seed = lambda: rand.integers(np.iinfo(int).max)

    # Validate axes.
    num_dim = len(x.shape)
    if axes is None:
        axes = range(num_dim)
    if np.isscalar(axes):
        axes = [axes]
    assert all(i in range(num_dim) for i in axes), 'invalid axis passed'

    # Draw axis and thickness.
    assert 0 < stride_min and stride_min <= stride_max, 'invalid strides'
    ind = tf.random.uniform(shape=[], minval=0, maxval=len(axes), dtype=tf.int32, seed=seed())
    ax = tf.gather(axes, ind)
    width = tf.gather(tf.shape(x), indices=ax)
    thick = tf.random.uniform(shape=[], minval=stride_min, maxval=stride_max, seed=seed())

    # Decide whether to downsample.
    assert 0 <= prob <= 1, f'{prob} not a probability'
    if prob < 1:
        rand_bit = tf.less(tf.random.uniform(shape=[], seed=seed()), prob)
        rand_not = tf.logical_not(rand_bit)
        thick = thick * tf.cast(rand_bit, thick.dtype) + tf.cast(rand_not, thick.dtype)

    # Resample.
    num_slice = tf.cast(width, thick.dtype) / thick + 0.5
    num_slice = tf.cast(num_slice, width.dtype)
    ind = tf.linspace(start=0, stop=width - 1, num=num_slice)
    ind = tf.cast(ind + 0.5, width.dtype)
    x = tf.gather(x, ind, axis=ax)
    if upsample:
        ind = tf.linspace(start=0, stop=tf.shape(x)[ax] - 1, num=width)
        ind = tf.cast(ind + 0.5, width.dtype)
        x = tf.gather(x, ind, axis=ax)

    return x


def gradient(y, axis=None):
    """
    Compute numerical gradients using second-order central differences with
    first-order one-sided forward/backward differences at the boundaries.

    Parameters:
        y: Tensor or NumPy array to compute gradients for.
        axis: Axes along which to compute gradients. Scalar or iterable.

    Returns:
        Tuple of gradients along each axis in the order these were specified, or a single tensor
        if `axis` is a scalar.

    Author:
        mu40

    If you find this function useful, please cite:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    # Tensor.
    if not tf.is_tensor(y) or y.dtype != tf.float32:
        y = tf.cast(y, tf.float32)

    # Axes.
    num_dim = len(y.shape)
    if axis is None:
        axis = range(num_dim)

    if hasattr(axis, '__iter__'):
        return tuple(gradient(y, axis=ax) for ax in axis)

    # Finite differences.
    forward = (axis, *range(axis), *range(axis + 1, num_dim))
    backward = (*range(1, axis + 1), 0, *range(axis + 1, num_dim))

    out = tf.transpose(y, perm=forward)
    beg = (out[1, ...] - out[0, ...])[None, ...]
    mid = 0.5 * (out[2:, ...] - out[:-2, ...])
    end = (out[-1, ...] - out[-2, ...])[None, ...]
    out = tf.concat((beg, mid, end), axis=0)

    return tf.transpose(out, perm=backward)


def jacobian(f, det=True, add_grid=True):
    """
    Compute the Jacobian matrix or its determinant at each location of a deformation or
    displacement field.

    Parameters:
        f: Deformation field as a Tensor or NumPy array of shape (..., N). The function considers
            the N axes before the last as spatial and any axes before that as batch dimensions.
        det:
            Return the determinant rather than the Jacobian matrix at each location.
        add_grid:
            Treat `f` as a displacement field instead of a deformation field.

    Returns:
        Jacobian matrix of shape (..., N, N) or its determinant of shape (...) at each voxel.

    Author:
        mu40

    If you find this function useful, please cite:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    # Tensor.
    if not tf.is_tensor(f) or f.dtype != tf.float32:
        f = tf.cast(f, tf.float32)

    # Validate shape.
    shape = f.shape
    num_dim = shape[-1]
    assert len(shape) > num_dim, f'field does not have {num_dim} spatial axes'

    # Compute Jacobian matrix J_ij=df_i/dx_j with respect to the N spatial axes before the last.
    # If f is a displacement field, we must add the coordinate grid to obtain the deformation
    # field F(x)=x+f(x). As dF(x)/dx=Id+df/dx, we could instead add an identity matrix to df/dx,
    # but this is less efficient on the GPU.
    axis = np.arange(num_dim) + len(shape) - num_dim - 1
    if add_grid:
        grid = (tf.range(shape[ax], dtype=tf.float32) for ax in axis)
        grid = tf.meshgrid(*grid, indexing='ij')
        f += tf.stack(grid, axis=-1)

    grad = gradient(f, axis=axis)
    jac = tf.stack(grad, axis=-1)

    return tf.linalg.det(jac) if det else jac


###############################################################################
# deprecation block
###############################################################################


def barycenter(x, axes=None, normalize=False, shift_center=False, dtype=tf.float32):
    """Compute barycenter along specified axes."""
    warnings.warn('Function nes.utils.barycenter is deprecated and will be '
                  'removed. Use ne.utils.barycenter instead.')
    return ne.utils.barycenter(x, axes, normalize, shift_center, dtype)


def apply_1d_filter(*args):
    '''
    Deprecate anytime after 12/01/2021

    See ne.utils.separable_conv()
    '''
    warnings.warn('nes.utils.apply_1d_filter has moved to ne.utils.separable_conv, '
                  'and will be deprecated here.')

    return ne.utils.separable_conv(*args)


def gaussian_blur(*args):
    """
    Deprecate anytime after 12/01/2021

    See nes.utils.gaussian_smoothing()

    TODO: add traceback.print_stack() ?
    """
    warnings.warn('Deprecation warning: renamed to gaussian_smoothing')
    return gaussian_smoothing(*args)


def whiten_1d(*args):
    """
    Deprecate anytime after 12/01/2021
    """
    warnings.warn('Deprecation warning: renamed to whiten')
    return ne.utils.whiten(*args)


def volume_centroid(vol):
    ''' compute the centroid of the input volume and return it in a (1,3) tensor '''
    # Make array of coordinates (each row contains three coordinates)
    volshape = tf.shape(vol)
    ii, jj, kk = tf.meshgrid(tf.range(volshape[0]),
                             tf.range(volshape[1]),
                             tf.range(volshape[2]), indexing='ij')
    coords = tf.stack([tf.reshape(ii, (-1,)),
                       tf.reshape(jj, (-1,)),
                       tf.reshape(kk, (-1,))], axis=-1)
    coords = tf.cast(coords, tf.float32)
    # Rearrange input into one vector per volume
    vol_flat = tf.reshape(vol, [-1, tf.math.reduce_prod(volshape), 1])
    total_mass = tf.reduce_sum(vol_flat, axis=1)  # Compute total mass for each volume
    centroid = tf.reduce_sum(vol_flat * coords, axis=1) / total_mass  # Compute center of mass
    return centroid


def pad_2d_image_spherically(img, pad_size=8, input_no_batch_dim=False):
    """
    pad parameterized 2d image based on the spherical positions of its vertices
    img: image to pad, whose shape is [batch_size, H, W, ...] or [H, W] for a single image
    """
    is_2d = is_nd(img, 2)
    img = expand_batch_dims_with_cond(img, is_2d, input_no_batch_dim)

    if pad_size > 0:
        # pad the north pole on top
        top = img[:, 1:pad_size + 1, ...]  # get top pad without the first row (reflect)
        top = flip(top, axis=1)  # flip upside down
        top = roll(top, get_shape(top, 2) // 2, axis=2)  # circularly shift by pi

        # similarly for the south pole on bottom
        bot = img[:, -pad_size - 1:-1, ...]
        bot = flip(bot, axis=1)
        bot = roll(bot, get_shape(bot, 2) // 2, axis=2)

        # concatenate top and bottom before padding left and right
        img2 = concat((top, img, bot), axis=1)

        # pad left to right and right to left (wrap)
        left = img2[:, :, 0:pad_size, ...]
        right = img2[:, :, -pad_size:, ...]
        img3 = concat((right, img2, left), axis=2)
    else:
        img3 = img

    img3 = squeeze_with_cond(img3, is_2d, input_no_batch_dim)

    return img3


def unpad_2d_image(img, pad_size=0, input_no_batch_dim=False):
    """
    extract the original image from the padded image
    img: image to unpad, whose shape is [batch_size, H, W, ...]
    """
    is_2d = is_nd(img, 2)
    img = expand_batch_dims_with_cond(img, is_2d, input_no_batch_dim)

    if pad_size > 0:
        img = img[:, pad_size:-pad_size, pad_size:-pad_size, ...]

    img = squeeze_with_cond(img, is_2d, input_no_batch_dim)
    return img


# wrap conditional functions compatible with both numpy and tf for padding and unpadding functions
def expand_batch_dims_with_cond(data, is_2d, input_no_batch_dim):
    if tf.is_tensor(data):
        is_2d = tf.cast(is_2d, tf.bool)
        input_no_batch_dim = tf.cast(input_no_batch_dim, tf.bool)
        cond = tf.logical_or(is_2d, input_no_batch_dim)
        data = tf.cond(cond, lambda: expand_batch_dims(data), lambda: data)
    else:
        if is_2d or input_no_batch_dim:
            data = expand_batch_dims(data)
    return data


def squeeze_with_cond(data, is_2d, input_no_batch_dim):
    if tf.is_tensor(data):
        is_2d = tf.cast(is_2d, tf.bool)
        input_no_batch_dim = tf.cast(input_no_batch_dim, tf.bool)
        cond = tf.logical_or(is_2d, input_no_batch_dim)
        data = tf.cond(cond, lambda: squeeze(data), lambda: data)
    else:
        if is_2d or input_no_batch_dim:
            data = squeeze(data)
    return data


# wrap some basic functions so they are compatible with both numpy and tf
def get_shape(data, axis=None):
    if tf.is_tensor(data):
        if axis is None:
            return tf.shape(data)
        else:
            return tf.shape(data)[axis]
    else:
        if axis is None:
            return data.shape
        else:
            return data.shape[axis]


def concat(data, axis):
    if tf.is_tensor(data[0]):
        return tf.concat(data, axis)
    else:
        return np.concatenate(data, axis)


def flip(data, axis):
    if tf.is_tensor(data):
        if not (isinstance(axis, list) or isinstance(axis, tuple)):
            axis = [axis]
        return tf.reverse(data, axis)
    else:
        return np.flip(data, axis)


def roll(data, shift, axis):
    if tf.is_tensor(data):
        return tf.roll(data, shift, axis)
    else:
        return np.roll(data, shift, axis)


def squeeze(data):
    if tf.is_tensor(data):
        return tf.squeeze(data)
    else:
        return np.squeeze(data)


def expand_batch_dims(data):
    if tf.is_tensor(data):
        return tf.expand_dims(data, 0)
    else:
        return data[np.newaxis, ...]


def is_nd(data, n):
    if tf.is_tensor(data):
        return tf.cond(tf.cast(tf.rank(data) == n, tf.bool),
                       lambda: tf.constant(True, dtype=tf.bool),
                       lambda: tf.constant(False, dtype=tf.bool))
    else:
        if data.ndim == n:
            return True
        else:
            return False


def model_size(model):
    """
    compute the comulative size of all output tensors in a model
    """

    total_size = 0
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            total_size += model_size(layer)
        else:
            outputs = layer.output
            if type(outputs) is not list:
                outputs = [outputs]
            for output in outputs:
                total_size += np.prod(np.array(output.get_shape().as_list()[1:]))

    return total_size


def spherical_sin(H, W, eps=1e-3):
    # the sin of latitude assumes the first and last element hits the poles
    # i.e. sin(0) and sin(pi), so that it can be padded at the north
    # and south poles properly by "reflection"
    rg = tf.transpose(tf.range(0, H, dtype=tf.float32))
    rg = tf.math.divide_no_nan(rg, to_tensor(H - 1))
    rg = tf.math.multiply(rg, to_tensor(np.pi))
    sin_lat = tf.math.sin(rg)
    # sin_lat = tf.math.sin(to_tensor(np.arange(0, H).T / (H - 1) * np.pi))
    # remove singularity near the two poles by setting the minimum to (positive) eps
    sin_lat = tf.where(sin_lat >= eps, sin_lat, eps)
    # repeat on longitude, forming the sin matrix
    S = tf.expand_dims(sin_lat, 1) * tf.ones((1, W))
    return S


def to_tensor(data, d_type=tf.float32):
    if tf.is_tensor(data):
        if data.dtype is not d_type:
            data = tf.cast(data, d_type)
        return data
    else:
        return tf.convert_to_tensor(data, d_type)
