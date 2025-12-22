# core python imports
from itertools import cycle

# third party imports
import numpy as np
import scipy.ndimage

# local imports
import neurite as ne


def complex2channels(x):
    """
    complex to channel along new (last) dimension
    (output has an extra dimension)
    """
    return np.stack([np.real(x), np.imag(x)], -1)


def rotate_complex(x, a):
    """
    rotate complex data (x is complex, s is shift)
    """
    r = scipy.ndimage.rotate(np.real(x), a)
    c = scipy.ndimage.rotate(np.imag(x), a)
    return r + 1j * c


def shift_complex(x, s):
    """
    shift complex data (x is complex, s is shift)
    """

    r = scipy.ndimage.interpolation.shift(np.real(x), s)
    c = scipy.ndimage.interpolation.shift(np.imag(x), s)
    return r + 1j * c


def kline_motion_gen(xvols,
                     batch_size=2,
                     rot_range=[-15, 15],
                     tr_range=[-5, 5],
                     nb_changes=3,
                     line_patience=1,  # smallest line to make change on
                     slice_ids=None,
                     seed=None,
                     verbose=False):
    """
    simulate motion in line-based kspace aquisition of images
    given an image, choose k-line places of "motion" --> fft each 
        motioned image --> combine fft base line right klines.
    """

    if seed is not None:
        np.random.seed(seed)

    # get complex
    if xvols.shape[-1] == 2:
        xvols_complex = xvols[..., 0] + 1j * xvols[..., 1]
    else:
        xvols_complex = xvols[..., 0] + 1j * xvols[..., 0] * 0

    # if slice ids are given
    if slice_ids is not None:
        slice_ids_iter = cycle(slice_ids)

    # go through selection
    while True:
        batch_in = []
        batch_out = []

        for _ in range(batch_size):

            if slice_ids is None:
                sidx = np.random.randint(0, xvols.shape[0])

            else:
                sidx = next(slice_ids_iter)

            xvolc = xvols_complex[sidx, ...]
            xvol_fft_0 = np.fft.fft2(xvolc, axes=(0, 1))

            xvol_fft = np.zeros(xvolc.shape, dtype=xvol_fft_0.dtype)

            rng = list(range(line_patience, xvolc.shape[0]))
            idx = np.random.choice(rng, size=nb_changes, replace=False)

            choices = np.sort(idx).tolist()
            choices.append(xvolc.shape[0])

            if verbose:
                print('choice idx:', choices)

            xvol_fft[:choices[0], ...] = xvol_fft_0[:choices[0], ...]

            zc = choices[0]
            for choice in choices[1:]:
                print('WARNING: NOT KEEPING SAME SIZE IN IMROTATE. SO ROTATION IS ALSO TRANSLATION')
                xvolc_rotc = rotate_complex(xvolc, np.random.uniform(*rot_range))
                xvolc_tr = shift_complex(xvolc, np.random.uniform(*tr_range, size=(2, 1)))
                xvol_fft[zc:choice, ...] = np.fft.fft2(xvolc_tr, axes=(0, 1))[zc:choice, ...]
                zc = choice

            batch_in.append(complex2channels(xvol_fft))
            batch_out.append(complex2channels(xvol_fft_0))

        yield np.stack(batch_in, 0), np.stack(batch_out, 0)


def load_mnist(pad=2, sel=None, imag_channel=False):
    """
    load mnist dataset with some options

    Parameters:
        pad: is amount to pad along each dimension. default of 2 to get data to 32x32
        sel: whether to select and return data with a label (digit)
        imag_channel: add 0 imaginary channel along last dimension

    Returns:
        (x_train, y_train), (x_test, y_test)
        x_train is [nb_elements, 28+pad*2, 28+pad*2, 1+imag_channel]
    """

    from keras.datasets import mnist

    pad_tuple = ((0, 0), (pad, pad), (pad, pad), (0, 0))

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[..., np.newaxis].astype('float') / 255
    x_train = np.pad(x_train, pad_tuple, 'constant')
    x_test = x_test[..., np.newaxis].astype('float') / 255
    x_test = np.pad(x_test, pad_tuple, 'constant')

    if sel is not None:
        x_train = x_train[y_train == sel, ...]
        x_test = x_test[y_test == sel, ...]

    if imag_channel:
        x_train = np.concatenate([x_train, x_train * 0], -1)
        x_test = np.concatenate([x_test, x_test * 0], -1)

    return (x_train, y_train), (x_test, y_test)


def load_knee(path, val_split=0, decim_factor=2, seed=0):
    """
    path e.g.: /nfs02/data/processed_nyu/NYU_training_Biograph.npy

    val_split whether to split some amount of validation in [0, 1)
    decim_factor: decimate (to downsample)
    seed: seed to randomization (when using val split)
    """
    xvols = np.load(path)

    if decim_factor is not None:
        xvols = xvols[:, ::decim_factor, ::decim_factor, :]

    if seed is not None:
        np.random.seed(seed)

    if val_split > 0:
        assert val_split < 1, 'val_split has to be in [0,1]'

        idx = list(range(0, xvols.shape[0]))
        np.random.shuffle(idx)

        nb_val = int(val_split * xvols.shape[0])
        xvals = xvols[idx[:nb_val], ...]
        xvols = xvols[idx[nb_val:], ...]

    else:
        xvals = np.zeros((0, *xvols.shape[1:]))

    return xvols, xvals


def kline_vis_pred_sample(sample, model=None, vis_mid_zoom=False):
    """
    visualize sample and prediction for kline experiment

    sample: should have input (motioned image), output (true image)
    """

    # predict via model
    if model is not None:
        pred = model.predict(sample[0])

    slices = [sample[1][0, ...], sample[0][0, ...]]

    if model is not None:
        slices += [pred[0, ...]]

    slices = [np.fft.ifft2(f[..., 0] + 1j * f[..., 1]) for f in slices]

    slices_mae = [np.abs(f - slices[0]).mean() for f in slices]
    titles = ['true img', 'input']

    if model is not None:
        titles += ['pred']

    titles = ['%s %f' % (titles[f], slices_mae[f]) for f in range(len(slices))]

    ne.plot.slices(slices, titles=titles, cmaps=['gray'], do_colorbars=True)

    if vis_mid_zoom:
        c = [f // 2 for f in slices[0].shape[:2]]
        d = [f // 5 for f in slices[0].shape[:2]]

        zoom_slices = [f[c[0] - d[0]:c[0] + d[0], c[1] - d[0]:c[1] + d[0]] for f in slices]
        ne.plot.slices(zoom_slices, titles=titles, cmaps=['gray'], do_colorbars=True)


def im2kspace(im, ratio=1, sample_type='linear', complex_out=False):
    """
    image to k_space via numpy

    Parameters:
        im: image to be transformed to k-space
        ratio (0.25): the ratio of (k-space) info to keep
        sample_type ('linear'): how to randomly sample. 'linear' or 'uniform' currently available
        complex_out (False): whether to output complex k_space, or stack into a [*im.shape, 2] array

    based on code from Cagla Bahadir
    """

    # inputs
    height = im.shape[0]

    # get k-space
    k_space = np.fft.fft2(im)
    k_space = np.fft.fftshift(k_space)

    # if under-sampling
    if ratio < 1:
        # create sampling matrix for uniform k-space rows
        if sample_type == 'linear':
            sample = np.zeros(im.shape)
            sample_idx = np.linspace(0, height - 1,
                                     num=ratio * height,
                                     endpoint=True,
                                     retstep=False,
                                     dtype=np.int16)
            sample[sample_idx, :] = 1

        elif sample_type == 'uniform':
            sample = np.random.choice([True, False], im.shape, p=[ratio, 1 - (ratio)]).astype('int')

        else:
            raise Exception('reconstruction: Unknown sample method')

        k_space = k_space * sample

    if complex_out:
        return k_space
    else:
        return np.stack((k_space.real, k_space.imag), 2)


def fft_layer(x):
    a = tf.complex(x, 0 * x)
    b = tf.fft2d(a)
    return K.concatenate([tf.real(b), tf.imag(b)], axis=1)


def ifft_layer(x):
    b = tf.ifft2d(x)
    return tf.real(b)
