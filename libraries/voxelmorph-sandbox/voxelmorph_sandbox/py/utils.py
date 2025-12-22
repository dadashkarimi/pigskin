"""
General python utilities.

Please code tensorflow/pytorch/jax utilities in their own folders
"""


# internal python imports
import sys
import warnings

# third party imports
import numpy as np
import scipy


# local (our) imports
import voxelmorph as vxm
import pystrum.pynd.ndutils as nd
import neurite_sandbox as nes
import neurite as ne


def jacobian_determinant(disp):
    """
    see vxm.py.utils.jacobian_determinant
    """
    print('jacobian_determinant has been moved to vxm.py.utils and will be deprecated from vxms',
          file=sys.stderr)
    return vxm.py.utils.jacobian_determinant(disp)


def locations_to_binary_vol(locs, volshape=None):
    """
    Given a list of locations (e.g. for surface points), create a binary volume with at the
    locations of those points

    Parameters:
        locs: locations in NxD array
        volshape: the shape of the destination volume

    Returns:
        binary volume
    """
    print('Warning: *VERY* experimental function', file=sys.stderr)

    # come input checking
    if volshape is None:
        volshape = np.ceil(np.max(locs)).astype(int)

    # prepare volume
    volume = np.zeros(volshape)
    loc_ints = np.round(locs).astype('int')

    # catch points beyond volume boundaries
    if np.any(loc_ints >= volshape) or np.any(loc_ints < 0):
        print('Found points beyond volume edges', file=sys.stderr)

    # clip surface points
    loc_ints = np.maximum(loc_ints, 0)
    loc_ints = np.minimum(loc_ints, [f - 1 for f in volshape])

    # binarize volume
    if len(volshape) == 3:
        volume[loc_ints[:, 0], loc_ints[:, 1], loc_ints[:, 2]] = 1
    else:
        volume[loc_ints[:, 0], loc_ints[:, 1]] = 1

    return volume


def remove_labels(vol, keep_labels=None, discard_labels=None, fill_val=0):
    """
    removes pixels/voxels with unwanted labels from a segmentation map

    Parameters:
        vol: the nd volume to be processed
        keep_labels: list of the labels to keep. must provide keep_labels xor discard_labels.
        discard_labels: list of labels to discard. must provide discard_labels xor keep_labels.
        fill_val: value (label) to fill in for locations of unwanted labels

    Returns:
        a copy of vol with unwanted values filled with fill_val
    """
    assert discard_labels is not None or keep_labels is not None, \
        'keep or unwanted labels needs to be provided'
    assert not (discard_labels is None and keep_labels is None), \
        'only one of keep or unwanted labels should be provided, not both'

    # existing labels in volume
    uniq = np.unique(vol).tolist()

    # compute which labels to keep
    if keep_labels is None:
        keep_labels = [f for f in uniq if f not in discard_labels]

    ovol = vol.copy()
    for u in uniq:
        if u not in keep_labels:
            ovol[np.where(vol == u)] = fill_val

    return ovol


def invert_warp(warp, interp_method='linear'):
    """
    Invert a warp using non-grid interpolation.

    Args:
        warp: numpy array of size [*vol_shape, ndims].
            example: [100, 100, 2]
        interp_method (str, optional): interpolation method.
            options available in scipy.interpolate.griddata method argument

    Returns:
        numpy array: inverse of this warp of same size

    Example:
        # simulate some data
        vol_shape = [100, 100]
        warp = np.random.normal(loc=0, scale=50, size=vol_shape + [2])
        warp = scipy.ndimage.gaussian_filter(warp, sigma=5)

        # invert warp
        inv_warp = invert_warp(warp)

        # visualize
        warp_tf = tf.convert_to_tensor(warp, tf.float32)
        warp_inv_tf = tf.convert_to_tensor(inv_warp, tf.float32)
        comp = vxm.utils.compose([warp_tf, warp_inv_tf]).numpy()
        ne.plot.flow([warp, inv_warp, comp])

    Author: adalca
    """

    warnings.warn("invert_warp is hyper-experimental")

    ndim = warp.shape[-1]
    volshape = warp.shape[:-1]

    # reshape warp for griddata interpolation
    warp = warp.reshape(-1, ndim)  # V x D

    # get grid and reshape to V x D
    grid = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid, -1).reshape(-1, ndim)

    # get off-grid inverse field
    invwarp_loc = grid + warp  # V x D
    invwarp_val = -warp        # V x D

    # interpolate to grid
    interp_fn = lambda x: scipy.interpolate.griddata(invwarp_loc, x, grid, method=interp_method)
    inv_warp_lst = [interp_fn(invwarp_val[..., d]) for d in range(ndim)]
    inv_warp = np.stack(inv_warp_lst, -1).reshape(volshape + (ndim,))

    # celebrate
    return inv_warp


def fit_warp_to_svf(warp,
                    nb_steps=5,
                    iters=100,
                    min_delta=1e-5,
                    lr=0.1,
                    init='warp',
                    verbose=True):
    """ 
    Experimental: Get an SVF to a warp

    Args:
        warp (np array): warp of size [*vol_shape, ndims]
        nb_steps (int, optional): Number of integral scaling and squaring steps
        iters (int, optional): number of iterations to fit
        min_delta ([type], optional): minimum delta for convergence
        lr (float, optional): learning rate (step size). Defaults to 0.1.
        init (str, optional): initialization strategy, 'warp' of 'jecwt'
        verbose (bool, optional): Defaults to True.

    Returns:
        [type]: [description]

    Author: adalca
    """

    import tensorflow as tf
    import tensorflow.keras.layers as KL

    inp = KL.Input((1,))
    vel_layer = ne.layers.LocalParamWithInput(shape=warp.shape)
    vel_tensor = vel_layer(inp)
    disp_tensor = vxm.layers.VecInt(int_steps=nb_steps)(vel_tensor)
    model = tf.keras.Model(inp, disp_tensor)

    # initialize the velocity field
    if init == 'warp':  # with a warp
        vel_layer.set_weights([warp])
    else:  # with a jacobian trick
        jac = jacobian_determinant(warp)
        jac3 = np.stack([jac] * warp.shape[-1], -1)
        init_wts = 1.02 * warp + (1 - jac3) * 0.05
        vel_layer.set_weights([init_wts])

    # compile and run model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='mse')
    callback = tf.keras.callbacks.EarlyStopping('loss', min_delta=min_delta)
    zero = np.zeros((1,))
    warpk = warp[np.newaxis, ...]
    hist = model.fit(zero, warpk,
                     epochs=iters,
                     verbose=0,
                     callbacks=[callback])

    if verbose:
        nes.utils.plot_keras_hist_losses(hist)

    return vel_layer.get_weights()[0]


def invert_warp_via_velocity(warp,
                             nb_steps=5,
                             iters=100,
                             **kwargs):
    """
    Experimental invert_warp algorithm:
    Fit a velocity field to displacement field, then negate and integrate.

    This starts being more useful than scipy griddata when you have large volumes.
    This might change if tf implements a griddata, but owuld likey not be easily    differentiable?

    Searched quite a bit on how we could do this more principled (quicker approx to vel from disp):
    - Discussion: https://itk.org/pipermail/insight-users/2010-September/037977.html
    - maybe use tf.linalg.expm and tf.linalg.logm (matrix logarim) in an effort to have a
        scale-and-square equivalent of "logarithmic map", but couldn't fully understand
        how possible this is
    - https://en.wikipedia.org/wiki/Derivative_of_the_exponential_map

    TODO:
    - think of implementing this in a layer, e.g. https://github.com/cvxgrp/cvxpylayers

    Args:
        warp (np array): warp of size [*vol_shape, ndims]
        nb_steps (int, optional): Number of integral scaling and squaring steps
        iters (int, optional): number of iterations to fit
        kwargs for warp_to_svf_fit

    Returns:
        approximate inverse warp of same size as warp

    Author: adalca
    """

    if iters > 0:
        # do it via a model.
        # TODO: Should change this to a normal optimization loop
        vel = fit_warp_to_svf(warp,
                              nb_steps=nb_steps,
                              iters=iters,
                              **kwargs)

        vel = vel
    else:
        vel = warp

    # integrate the negative velocity to get inverse
    vel_tensor = tf.convert_to_tensor(vel, tf.float32)
    return vxm.utils.integrate_vec(-vel_tensor, nb_steps=nb_steps).numpy()
