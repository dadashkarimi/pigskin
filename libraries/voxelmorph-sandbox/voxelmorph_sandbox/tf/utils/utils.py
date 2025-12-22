"""
General tf utilities for VoxelMorph (sandbox)

** If you find yourself developing a ton of utilities for some specific sub-task 
(e.g. data processing), it might be best to start a new file (e.g. data.py)

** to move functions, keep a header here with a deprication warning
"""

# internal python imports
import warnings

# third party imports
import numpy as np
from numpy.lib.twodim_base import mask_indices
import tensorflow as tf
import tensorflow.keras.backend as K

# local (our) imports
import voxelmorph as vxm
from .. import networks
import neurite as ne
import neurite_sandbox as nes


def hard_dice(vol1, vol2, labels):
    """
    TF function to compute the Dice score for several labels between two tensors.
    TODO: compare and merge with hard option in ne.metrics.Dice()

    Parameters:
        vol1 and vol2: Tensors of the samve volume size
        labels: list of labels (ints) we care about

    Returns:
        Tensor of size nb_labels -- Dice score for all labels
    """

    dicem = [None] * len(labels)
    for idx, lab in enumerate(labels):

        # extract binary volumes
        vol1l = tf.cast(tf.equal(vol1, lab), 'float32')
        vol2l = tf.cast(tf.equal(vol2, lab), 'float32')

        # compute Dice
        top = 2 * K.sum(vol1l * vol2l)
        bottom = K.sum(vol1l) + K.sum(vol2l)
        bottom = K.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    # stack return
    return tf.stack(dicem)


def get_hard_dice_function(volshape, labels):
    """
    Return a keras function that can be called with numpy volumes.
    This should not be necessary after tf 2.0+

    Parameters:
        volshape: the shape of the expected volumes
        labels: list of labels (ints)

    Returns:
        a function fn, such that dice_scores = fn(numpy_array)
    """
    z_inp1 = tf.placeholder(tf.float32, volshape)
    z_inp2 = tf.placeholder(tf.float32, volshape)
    z_out = hard_dice(z_inp1, z_inp2, labels)
    return K.function([z_inp1, z_inp2], [z_out])


def decompose_warp(
    shift, weights=None, shift_center=True, last_row=False,
    return_mat=True, return_shift=False, return_diff=False,
):
    """Decompose an N-dimensional displacement field.

    Decompose an N-dimensional displacement field into an affine and a
    residual deformable component, in an ordinary or weighted least-squares
    (LS) sense. The function supports inputs with or without batch dimensions.

    Arguments:
        shift: Array-like displacement field to decompose, of shape
            (*batch, *space, N), where space is a tuple of length N, and batch
            is a tuple of any length.
        weights: Optional array-like weights at each voxel for weighted LS, of
            shape (*batch, *space, 1). The last dimension can be omitted.
        shift_center: Compute the affine matrix in a centered coordinate frame.
        last_row: Return a full matrix with N + 1 rows instead of omitting the
            last row, which is always ``(*[0] * N, 1``.
        return_mat: Return the fitted affine transformation matrix of shape
            (*batch, *space, N, N + 1) or (*batch, *space, N, N + 1),
            depending on `last_row`.
        return_shift: Return the displacement field representing the affine
            component. Will be of the same shape as the input field.
        return_diff: Return the left-over displacement that the affine matrix
            cannot encompass. Will have the same shape as the input field.

    Returns:
        The fitted affine transformation and decomposed displacements as
        specified above. The output type will be identical to the input type.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    # Types and shapes.
    out_type = compute_type = tf.float32
    if shift.dtype != compute_type:
        out_type = shift.dtype
        shift = tf.cast(shift, compute_type)
    num_dim = shift.shape[-1]
    shape = tf.shape(shift)
    shape_batch = shape[:-(num_dim + 1)]
    shape_space = shift.shape[-(num_dim + 1):-1]

    # Coordinate grid.
    grid = (np.arange(x, dtype=compute_type.as_numpy_dtype) for x in shape_space)
    if shift_center:
        grid = (g - (v - 1) / 2 for g, v in zip(grid, shape_space))
    grid = [np.ravel(x) for x in np.meshgrid(*grid, indexing='ij')]
    xyz = tf.stack(grid, axis=-1)

    # Flattened tensors.
    shape_flat = tf.concat((shape_batch, (-1, num_dim)), axis=0)
    shift = tf.reshape(shift, shape_flat)
    loc = xyz + shift

    # Weights.
    if weights is not None:
        if weights.dtype != compute_type:
            weights = tf.cast(weights, compute_type)
        weights = tf.reshape(weights, shape=shape_flat[:-1])

    # Least squares.
    aff = vxm.utils.fit_affine(x_source=loc, x_target=xyz, weights=weights)
    aff_trans = tf.linalg.matrix_transpose(aff)
    aff_loc = xyz @ aff_trans[..., :-1, :] + aff_trans[..., -1:, :]
    aff_shift = aff_loc - xyz
    aff_diff = shift - aff_shift
    aff_shift = tf.reshape(aff_shift, shape)
    aff_diff = tf.reshape(aff_diff, shape)

    # Transformation matrix.
    if last_row:
        zeros_shape = tf.concat((shape_batch, (1, num_dim)), axis=0)
        zeros = tf.zeros(zeros_shape, dtype=compute_type)
        one_shape = tf.concat((shape_batch, (1, 1)), axis=0)
        one = tf.ones(one_shape, dtype=compute_type)
        last_row = tf.concat((zeros, one), axis=-1)
        aff = tf.concat((aff, last_row), axis=0)

    # Outputs.
    out = []
    if return_mat:
        out.append(aff)
    if return_shift:
        out.append(aff_shift)
    if return_diff:
        out.append(aff_diff)

    if out_type != compute_type:
        out = [tf.cast(x, out_type) for x in out]
    return out[0] if len(out) == 1 else out


def instance_affine_register(vol1, vol2,
                             loss='mse',
                             max_iter=10000):
    """
    Affinely register two volumes using an instance trainer.
    TODO: add more callback parameters are input parameters here

    Note: a scale-space version of this is available and in development.

    Parameters:
        vol1 and vol2: two volumes, *not* in keras format.
        loss: the loss to use in the instance trainer. Default: 'mse'
        max_iter: the maximum number of iterations
    """
    warnings.warn("deprecated", DeprecationWarning)

    volshape = list(vol1.shape)

    # prepare models
    model = networks.InstanceAffine(volshape)
    affine_out_model = model.get_affine_model(return_dense=False)

    # prepare callbacks for smartly adapting to optimization
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                    factor=0.1,
                                                    patience=10,
                                                    cooldown=5,
                                                    min_lr=1e-7)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                 patience=20)

    # compile
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-1), loss=loss)

    # train
    v1 = vol1[np.newaxis, ..., np.newaxis]
    v2 = vol2[np.newaxis, ..., np.newaxis]
    hist = model.fit(v1, v2, epochs=max_iter, verbose=0, callbacks=[reducelr, earlystop])

    # warp and return
    v1_warped = model.predict(v1)
    affine = affine_out_model.predict(v1)
    return v1_warped.squeeze(), affine


def transform_grid(warp):
    ''' warp is vol_shape + [len(vol_shape)], so no batch size '''
    grid = ne.utils.volshape_to_ndgrid(warp.shape[:-1])
    return vxm.utils.transform(warp, tf.cast(tf.stack(grid, -1), tf.float32))


def bbox3D(img):
    ''' compute the bounding box of the non-zero region of an image
    and return it as a matrix of (augmented) column vectors of the corners '''
    img = tf.cast(img, tf.bool)
    row = tf.math.reduce_any(img, axis=(1, 2))
    col = tf.math.reduce_any(img, axis=(0, 2))
    sl = tf.math.reduce_any(img, axis=(0, 1))
    rwhere = tf.where(row)
    cwhere = tf.where(col)
    swhere = tf.where(sl)
    rmin = rwhere[0, 0]
    rmax = rwhere[-1, 0]
    cmin = cwhere[0, 0]
    cmax = cwhere[-1, 0]
    smin = swhere[0, 0]
    smax = swhere[-1, 0]
    bbox = tf.convert_to_tensor([
        [rmin, cmin, smin, 1],
        [rmin, cmin, smax, 1],
        [rmin, cmax, smin, 1],
        [rmin, cmax, smax, 1],
        [rmax, cmin, smin, 1],
        [rmax, cmin, smax, 1],
        [rmax, cmax, smin, 1],
        [rmax, cmax, smax, 1]
    ], dtype=tf.float32)
    return tf.transpose(bbox)


def bboxND(img):
    ''' compute the bounding box of the non-zero region of an image
    and return it as a matrix of (augmented) column vectors of the corners '''
    img = tf.cast(img, tf.bool)
    ndims = 2 if len(img.shape) < 3 else 3

    if ndims == 3:
        row = tf.math.reduce_any(img, axis=(1, 2))
        col = tf.math.reduce_any(img, axis=(0, 2))
        sl = tf.math.reduce_any(img, axis=(0, 1))
        rwhere = tf.where(row)
        cwhere = tf.where(col)
        swhere = tf.where(sl)
        rmin = rwhere[0, 0]
        rmax = rwhere[-1, 0]
        cmin = cwhere[0, 0]
        cmax = cwhere[-1, 0]
        smin = swhere[0, 0]
        smax = swhere[-1, 0]
        bbox = tf.convert_to_tensor([
            [rmin, cmin, smin, 1],
            [rmin, cmin, smax, 1],
            [rmin, cmax, smin, 1],
            [rmin, cmax, smax, 1],
            [rmax, cmin, smin, 1],
            [rmax, cmin, smax, 1],
            [rmax, cmax, smin, 1],
            [rmax, cmax, smax, 1]
        ], dtype=tf.float32)
    else:
        row = tf.math.reduce_any(img, axis=(1))
        col = tf.math.reduce_any(img, axis=(0))
        rwhere = tf.where(row)
        cwhere = tf.where(col)
        rmin = rwhere[0, 0]
        rmax = rwhere[-1, 0]
        cmin = cwhere[0, 0]
        cmax = cwhere[-1, 0]
        bbox = tf.convert_to_tensor([
            [rmin, cmin, 1],
            [rmin, cmax, 1],
            [rmax, cmin, 1],
            [rmax, cmax, 1]
        ], dtype=tf.float32)

    return tf.transpose(bbox)


def check_batch_is_invertible(mat_batch, ndims):
    # check a batch of affine matrices to see if they are all invertible

    def _check_single_mat(mat):
        # remove trans col and pick out square submat
        mat = tf.reshape(mat, (ndims, ndims + 1))
        mat = tf.reshape(mat[:, :-1], (ndims, ndims)) + tf.eye(ndims)
        nes.utils.print_if_true(
            nes.utils.is_not_invertible(mat),
            mat,
            output_stream='file://full_affine.tfp',
            summarize=-1)
        return mat

    tf.map_fn(_check_single_mat, mat_batch, fn_output_signature=tf.float32)
    return mat_batch


def random_warp(vol_shape, amp_mean=0, amp_stddev=1, smooth_stddev=1):
    """ Generate a random deformation field

    Args:
        vol_shape (list): size of the volume, e.g. [100, 100, 100]
        amp_mean (int, optional): mean displacement amplitude. Defaults to 0.
        amp_stddev (int, optional): standard deviation of amplitude. Defaults to 1.
        smooth_stddev (int, optional): smoothness standard deviation.. Defaults to 1.

    Returns:
        a warp field of size volshape + [ndim], e.g. [100, 100, 100, 3]
    """

    field_shape = list(vol_shape) + [len(vol_shape)]

    field_rnd = tf.random.normal(shape=field_shape, mean=amp_mean, stddev=amp_stddev)
    smoothed_field = nes.utils.gaussian_smoothing(field_rnd, smooth_stddev)

    return smoothed_field


def batch_random_warp(vol_shape, batch_size, amp_mean=0, amp_stddev=1, smooth_stddev=1):
    """ Generate random deformation fields in a batch

    Args:
        vol_shape (list): size of the volume, e.g. [100, 100, 100]
        batch_size (int): size of batch. e.g. 16
        amp_mean (int, optional): mean displacement amplitude. Defaults to 0.
        amp_stddev (int, optional): standard deviation of amplitude. Defaults to 1.
        smooth_stddev (int, optional): smoothness standard deviation.. Defaults to 1.

    Returns:
        batch of warp fields of size [batch_size] + volshape + [ndim], e.g. [16, 100, 100, 100, 3]
    """

    field_shape = [batch_size] + list(vol_shape) + [len(vol_shape)]

    field_rnd = tf.random.normal(shape=field_shape, mean=amp_mean, stddev=amp_stddev)
    smoothed_field = nes.utils.batch_gaussian_smoothing(field_rnd, smooth_stddev)

    return smoothed_field


def random_diffeomorphic_warp(vol_shape,
                              nb_steps=5,
                              amp_mean=0,
                              amp_stddev=1,
                              smooth_stddev=1,
                              return_inverse=False):
    """ Generate a random diffeomorphic deformation field

    Args:
        vol_shape (list): size of the volume, e.g. [100, 100, 100]
        nb_steps (int): number of integration steps for diffeomorphic integration
        amp_mean (int, optional): mean displacement amplitude. Defaults to 0.
        amp_stddev (int, optional): standard deviation of amplitude. Defaults to 1.
        smooth_stddev (int, optional): smoothness standard deviation.. Defaults to 1.
        return_inverse (bool): whether to return the inverse field as well.

    Returns:
        a warp field of size volshape + [ndim], e.g. [100, 100, 100, 3]
        or
        a tuple of the warp field *and* the inverse of the same size

    Author: adalca
    """

    field_shape = list(vol_shape) + [len(vol_shape)]

    field_rnd = tf.random.normal(shape=field_shape, mean=amp_mean, stddev=amp_stddev)
    smoothed_svf = nes.utils.gaussian_smoothing(field_rnd, smooth_stddev)

    field = vxm.utils.integrate_vec(smoothed_svf, nb_steps=nb_steps)

    if return_inverse:
        inv_field = vxm.utils.integrate_vec(-smoothed_svf, nb_steps=nb_steps)
        return field, inv_field

    else:
        return field


def batch_random_affine(batch_size,
                        num_dim,
                        in_shape,
                        out_shape,
                        max_shift=0,
                        max_rotate=0,
                        max_scale=0,
                        max_shear=0,
                        axes_flip=False,
                        axes_swap=False,
                        seeds={},
                        id=0):
    from . import augment

    # Affine transform.
    parameters = tf.concat((
        tf.random.uniform(shape=(batch_size, num_dim),
                          minval=-max_shift, maxval=max_shift, seed=seeds.get('shift')),
        tf.random.uniform(shape=(batch_size, 1 if num_dim == 2 else 3),
                          minval=-max_rotate, maxval=max_rotate, seed=seeds.get('rotate')),
        tf.random.uniform(shape=(batch_size, num_dim),
                          minval=-max_scale, maxval=max_scale, seed=seeds.get('scale')),
        tf.random.uniform(shape=(batch_size, 1 if num_dim == 2 else 3),
                          minval=-max_shear, maxval=max_shear, seed=seeds.get('shear')),
    ), axis=-1)
    affine = vxm.utils.params_to_affine_matrix(
        parameters, deg=True, shift_scale=True, last_row=True, ndims=num_dim,
    )

    in_shape = np.array(in_shape)
    out_shape = np.array(out_shape)

    # Center of rotation.
    cen = np.eye(num_dim + 1, dtype=np.float32)
    cen[:num_dim, -1] = -0.5 * (out_shape - 1)
    trans = tf.matmul(np.linalg.inv(cen), tf.matmul(affine, cen), name=f'rot_cen_{id}')
    # Symmetric padding.
    shift = np.eye(num_dim + 1, dtype=np.float32)
    shift[:num_dim, -1] = np.round(0.5 * (in_shape - out_shape))
    trans = tf.matmul(shift, trans, name=f'out_cen_{id}')

    # if axes_flip:
    #     flip_draw = lambda x: tf.matmul(x, vxm.utils.draw_flip_matrix(
    #         out_shape, shift_center=False, seed=seeds.get('flip'),
    #     ))
    #     trans = KL.Lambda(flip_draw, name=f'flip_{id}')(trans)

    # if axes_swap:
    #     assert all(x == out_shape[0] for x in out_shape), 'output dimensions not all identical'
    #     swap_draw = lambda x: tf.matmul(
    #         x,
    #         vxm.utils.draw_swap_matrix(num_dim, seed=seeds.get('swap')),
    #     )
    #     trans = KL.Lambda(swap_draw, name=f'swap_{id}')(trans)

    return trans


def simulate_rigid_motion(image,
                          max_shift,
                          max_rotate,
                          idx_replace,
                          line_axis=0,
                          shuffle_lines=True,
                          return_lists=False):
    """ [Experimental] simulate rigid motion of a given image

    Args:
        image ([type]): the input image in keras-sizing [batch_size, ..., nb_feats]
        max_shift: the maximum shift in voxels per motion
        max_rotate: the maximum rotation in degrees per motion
        idx_replace ([type]): list of indices at which to replace the k-lines.
            TODO: allow option for this to be 'None' and do it internally given number of motions
        line_axis: axis within the volume dimensions
        shuffle_lines (bool, optional): whether to acquire lines in shuffled (random) order. 
            Defaults to False.
        return_lists (bool, optional): return a bunch of data, not just warped image.
            This output is not finalized, might change
            Defaults to False.

    Returns:
        image: warped image

    Author: adalca
    """
    im_shape = image.shape[1:-1]
    nb_dims = len(im_shape)
    batch_size = tf.shape(image)[0]
    assert line_axis >= 0, 'line_axis should be in [0, nb_dims)'
    assert line_axis < nb_dims, 'line_axis should be in [0, nb_dims)'
    kaxis = line_axis + 1

    nb_steps = len(idx_replace)

    # k-space of main image
    ksp = ne.layers.FFTShift()(ne.layers.FFT()(image))

    affines = []
    warped = []
    ksp_stack = [ksp]
    for _ in idx_replace:
        aff = batch_random_affine(batch_size,
                                  nb_dims,
                                  in_shape=im_shape,
                                  out_shape=im_shape,
                                  max_shift=max_shift,
                                  max_rotate=max_rotate,
                                  max_scale=0,
                                  max_shear=0)

        image_warp = vxm.layers.SpatialTransformer()([image, aff[:, :nb_dims, ...]])

        # warps
        ksp_warp = ne.layers.FFTShift()(ne.layers.FFT()(image_warp))

        # save data
        affines.append(aff)
        warped.append(image_warp)
        ksp_stack.append(ksp_warp)

    idx_start = [idx_replace[0] * 0] + idx_replace
    idx_end = idx_replace + [idx_replace[0] * 0 + ksp.shape[kaxis]]
    v1 = tf.logical_not(tf.sequence_mask(idx_start, ksp.shape[kaxis]))
    v2 = tf.sequence_mask(idx_end, ksp.shape[kaxis])
    masks = tf.squeeze(tf.cast(tf.logical_and(v1, v2), tf.complex64))  # nb_motion x S

    if shuffle_lines:
        masks = K.transpose(tf.random.shuffle(K.transpose(masks)))

    # reshape masks
    o = [1] * (len(im_shape) + 3)  # 3 for keras sizing + nb motion
    o[0] = masks.shape[0]
    o[kaxis + 1] = masks.shape[1]
    masks = tf.reshape(masks, o)  # [nb_motion, 1, ..., S, 1, ...]

    # form output k-space
    ksp_out = 0
    for i in range(nb_steps + 1):
        ksp_out += ksp_stack[i] * masks[i]

    # output image
    im_out = ne.layers.IFFT()(ne.layers.IFFTShift()(ksp_out))

    # prepare outputs
    if return_lists:
        return im_out, affines, warped, ksp_stack
    else:
        return im_out
