"""
tensorflow/keras utilities augmentation for the neuron project.


Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

# third party imports
import numpy as np
import tensorflow as tf
from keras import layers as KL

# local imports
import neurite_sandbox as nes
import neurite as ne

import voxelmorph as vxm
from .utils import spherical_sin, pad_2d_image_spherically


def add_outside_shapes(
        im,
        seg,
        labels_in,
        num_outside_shapes_to_add=8,
        outside_dist=4,
        zoom_factor=16,
        l2i=None,
        l2l=None,
        pct_skull_retained=.4,
        pct_air_retained=.9,
        return_l2l=False,
        return_l2i=False
):

    """
  add noise and random structure outside of labeled regions

  Parameters:
    im,                          - input image to add noise to (no batch or channels dimension yet)
    seg,                         - input segmentation (no batch or channels dimension yet,
                                   not onehotted)
    labels_in,                   - set of input labels
    num_outside_shapes_to_add=8, - number of exterior "skull" labels to add
    outside_dist=4,              - max distance in vox that noise/skull will bed added outside of labels
    zoom_factor=16,              - the spatial scale of the randomness. Bigger zoom means smaller
                                   blobs (im.shape must be integer divisible by this number)
    l2i=None,                    - nes.models.labels_to_image model (if None will be allocated
                                   internally)
    l2l=None,                    - nes.models.labels_to_labels model (if None will be allocated
                                   internally)
    pct_skull_retained=.4,       - amount of skull retained. Not really a pct, but closer to
                                   1 will keep more
    pct_air_retained=.9          - amount of air retained. Not really a pct, but closer
                                   to 1 will keep more
    """

    assert len(im.shape) == 3, 'add_outside_shape: input image must not have a channels or batch dim'
    assert len(seg.shape) == 3, 'add_outside_shape: input seg must not have a channels or batch dim'
    assert pct_skull_retained > 0, 'add_outside_shape: pct_skull_retained must be > 0'
    assert pct_air_retained > 0, 'add_outside_shape: pct_air_retained must be > 0'

    if l2l is None:
        l2l = nes.models.labels_to_labels(
            labels_in,
            shapes_num=num_outside_shapes_to_add,
            in_shape=im.shape,
            shapes_add=True
        )
    if l2i is None:
        # add outside shape labels into list for internal use
        nlabels = labels_in.max() + 1
        li = np.concatenate([labels_in,
                             np.arange(nlabels, nlabels + num_outside_shapes_to_add)])
        l2i = nes.models.labels_to_image(
            labels_in=li,
            labels_out=None,
            in_shape=im.shape,
            zero_background=1,
            noise_max=.2,
            noise_min=.1,
            warp_max=0
        )

    dtype = im.dtype
    seg = seg[..., tf.newaxis]

    mx = tf.reduce_max(im)
    im /= mx

    # synth a label map and associated image with shapes outside of current labels
    labels_with_outside = l2l(seg[tf.newaxis, ...])
    noise_skull, lwn = l2i(labels_with_outside)
    label_dtrans = nes.tf.utils.morphology_3d(tf.cast(seg > 0, dtype=tf.int32),
                                              1, 2 * outside_dist)[..., 0]

    # create some noisy "air" outside of the current labels
    outside_ring = tf.logical_and(label_dtrans > 0, label_dtrans <= outside_dist)
    noise_air = tf.cast(outside_ring, dtype) * tf.random.uniform(shape=seg.shape[:-1],
                                                                 minval=0, maxval=.1, dtype=dtype)

    rand_dtrans_skull_down = tf.random.uniform(
        shape=tuple(np.array(seg.shape[:-1]) // zoom_factor),
        minval=0, maxval=outside_dist / pct_skull_retained,
        dtype=dtype)[..., tf.newaxis]
    rand_dtrans_skull = ne.tf.utils.zoom(rand_dtrans_skull_down, zoom_factor)[..., 0]

    # air dtrans starts from -outside dist so it hugs the brain
    rand_dtrans_air_down = tf.random.uniform(
        shape=tuple(np.array(seg.shape[:-1]) // zoom_factor),
        minval=-outside_dist,
        maxval=outside_dist / pct_air_retained,
        dtype=dtype)[..., tf.newaxis]
    rand_dtrans_air = ne.tf.utils.zoom(rand_dtrans_air_down, zoom_factor)[..., 0]

    mask_skull = tf.logical_and(rand_dtrans_skull <= label_dtrans, outside_ring)
    mask_air = tf.logical_and(rand_dtrans_air <= label_dtrans, outside_ring)

    # mask the two noise images so that they occur near labels with variable thickness
    im_noise_skull = tf.multiply(noise_skull[0, ..., 0], tf.cast(mask_skull, dtype))
    im_noise_air = tf.multiply(noise_air, tf.cast(mask_air, dtype))
    im_ret1 = tf.add(im, im_noise_air)   # add skull stripping errors
    im_ret = mx * tf.add(im_noise_skull, im_ret1)    # add some non-brain intensities

    rets = [im_ret]
    if return_l2l:
        rets += [l2l]
    if return_l2i:
        rets += [l2i]
    if len(rets) == 1:
        rets = rets[0]   # backwards compatibility
    return rets


def flash_forward(PD, T1, T2star, alpha, TR, TE):
    PD = PD * np.exp(-TE / T2star)
    E1 = np.exp(-TR / T1)
    S = PD * np.sin(alpha) * (1 - E1) / (1 - np.cos(alpha) * E1)
    return S


# Network to simulate skull stripping artifacts
def simulate_extracerebral_artifacts(in_shape,
                                     left_cerebral_cortex_label,
                                     right_cerebral_cortex_label,
                                     outside_ring_label,
                                     thresh=0.1,
                                     fwhm_min=10,
                                     fwhm_max=20,
                                     noise_min=0.01,
                                     noise_max=1,
                                     gaussian_smooth_sigma=9,
                                     side_gaussian_smooth_sigma=27,
                                     label_dtrans_thresh=0.05,
                                     outside_thresh_min=0.3,
                                     outside_thresh_max=0.4,
                                     outside_prob=0.5,
                                     name='extra_artifacts',
                                     dtype=None,
                                     seed=None,
                                     return_all=False):

    assert 0 < noise_min <= noise_max, f'invalid noise-SD bounds {(noise_min, noise_max)}'

    # Inputs
    input_seg = KL.Input(shape=(*in_shape, 1), name='%s_input' % name)
    gt_seg = KL.Input(shape=(*in_shape, 1), name='%s_ground_truth' % name)
    model_inputs = [input_seg, gt_seg]

    if not input_seg.dtype.is_integer:
        input_seg = tf.cast(input_seg, tf.int32)
    if not gt_seg.dtype.is_integer:
        gt_seg = tf.cast(gt_seg, tf.int32)

    # Data types
    compute_type = tf.keras.mixed_precision.global_policy().compute_dtype
    compute_type = tf.dtypes.as_dtype(compute_type)
    dtype = compute_type if dtype is None else tf.dtypes.as_dtype(dtype)

    # Approximates distance to the brain and left/right cortex
    label_dtrans       = nes.tf.utils.batch_gaussian_smoothing(input_seg > 0, gaussian_smooth_sigma)
    label_dtrans_left  = nes.tf.utils.batch_gaussian_smoothing(input_seg == left_cerebral_cortex_label, side_gaussian_smooth_sigma)
    label_dtrans_right = nes.tf.utils.batch_gaussian_smoothing(input_seg == right_cerebral_cortex_label, side_gaussian_smooth_sigma)

    # Outside_ring is a binary mask surrounding the brain
    outside_ring = tf.logical_and(label_dtrans > label_dtrans_thresh, input_seg == 0)
    outside_ring_left  = tf.logical_and(label_dtrans_left > label_dtrans_right, outside_ring)
    outside_ring_right = tf.logical_and(label_dtrans_right >= label_dtrans_left, outside_ring)
    outside_ring = tf.cast(outside_ring, compute_type)
    outside_ring_left  = tf.cast(outside_ring_left, tf.int32)
    outside_ring_right = tf.cast(outside_ring_right, tf.int32)

    # Randomization
    rand = np.random.default_rng(seed)
    seed = lambda: rand.integers(np.iinfo(np.int32).max, dtype=np.int32)
    prop = lambda: dict(seed=seed(), dtype=compute_type)
    std_min = fwhm_min / 2.355
    std_max = fwhm_max / 2.355
    blur = lambda x: nes.utils.generative.random_blur_rescale(
        x, std_min=std_min, std_max=std_max, batched=True, rand=rand)

    # Create random blobs within the outside_ring
    noise_std = KL.Lambda(lambda x: tf.random.uniform(shape=(), minval=noise_min, maxval=noise_max, **prop()))(outside_ring)
    noise = tf.random.normal(tf.shape(input_seg), stddev=noise_std, **prop())
    noise = blur(noise)
    noise = noise * outside_ring
    noise = ne.utils.minmax_norm(noise)
    shapes = tf.cast(noise < thresh, tf.int32)
    shapes = (shapes * outside_ring_left * left_cerebral_cortex_label) + (shapes * outside_ring_right * right_cerebral_cortex_label)

    # Merge the original segmentation with the random blobs
    seg_with_blobs = input_seg + shapes

    # Randomly create a ring around the brain, outside of the random blobs
    out_p = KL.Lambda(lambda x: tf.random.uniform(shape=(), minval=0, maxval=1, **prop()))(seg_with_blobs)
    do_out = tf.cast(tf.math.less(out_p, outside_prob), compute_type)
    # either outside_thresh or 1 (make no ring), with 50% probability
    outside_thresh = KL.Lambda(lambda x: tf.random.uniform(shape=(), minval=outside_thresh_min, maxval=outside_thresh_max, **prop()))(seg_with_blobs)
    outside_thresh = do_out*outside_thresh + (1-do_out)*1
    outside_ring2 = tf.logical_and(label_dtrans > outside_thresh, seg_with_blobs == 0)
    outside_ring2 = tf.cast(outside_ring2, tf.int32)
    seg_with_blobs = seg_with_blobs + (outside_ring2 * outside_ring_label) 

    if not return_all:
        model_outputs = [seg_with_blobs, gt_seg]
    else:
        model_outputs = [input_seg, gt_seg, label_dtrans, outside_ring, noise_std,
                         noise, shapes, out_p, outside_thresh, outside_ring2, seg_with_blobs]

    return tf.keras.Model(inputs=model_inputs, outputs=model_outputs, name='%s_model' % name)

@tf.autograph.experimental.do_not_convert
def spherical_augment(img, types=[], noise_max=1.0, deform_min=0.0, deform_max=6.0, 
                      deform_sigma=8.0,
                      pad_size=0, is_deform_smooth=True, eps=1e-1, random_mag=True):
    """
    data augmentation for spherical images
    :param img: input image of shape [B, H, W, C]
    :param types: a list of augmentation types, current support: 'noise', 'deform'
    :param noise_max: maximum noise magnitude
    :param deform_min: maximum deformation magnitude
    :param deform_max: maximum deformation magnitude
    :param deform_sigma: gaussian smoothing sigma for deformation field
    :param pad_size: pad size for distortion correction
    :param is_deform_smooth: binary flag indicating whether to smooth the deformation field
    :param eps: the epsilon value for spherical sin function to prevent crazy division at poles
    :param random_mag: is the magnitude of augmentation random between 0 and max
    :return: augmented image
    """

    import tensorflow_addons as tfa

    if isinstance(img, (list, tuple)):
        was_list = True
    else:
        img = [img]
        was_list = False

    img0_shape = tf.shape(img[0])
    B, H, W = img0_shape[0], img0_shape[1], img0_shape[2]
    H_s, W_s = H - pad_size * 2, W - pad_size * 2
    S = spherical_sin(H_s, W_s, eps)
    S = tf.expand_dims(S, axis=0)
    S = pad_2d_image_spherically(S, pad_size=pad_size)

    def random_noise(shape, min_val, max_val):
        if random_mag:
            mag = tf.random.uniform([], min_val, max_val)
        else:
            mag = max_val
        return mag * tf.random.normal(shape)

    if 'noise' in types:  # different random noise is applied to different images in the list
        img = [x + random_noise(tf.shape(x), 0, noise_max) for x in img]

    def random_deform(shape):
        # generate random velocity field and convert to tensor
        warp = random_noise(shape, deform_min, deform_max)
        # smooth velocity field and compensate for magnitude reduction
        if is_deform_smooth:
            # determine filter size to be nearest even number to 3 sigma + 1
            f_size = tf.cast(tf.round(deform_sigma * 3 // 2) * 2 + 1, tf.int32)
            f_size_float = tf.cast(f_size, tf.float32)
            # gaussian filter and multiply by filter size
            # the loss of magnitude is due to the average of iid gaussian, where the sigma is
            # reduced by the factor of sqrt(N), where N is the number of samples
            # N = f_size^2, so the sigma should be corrected by f_size
            warp = tfa.image.gaussian_filter2d(warp, filter_shape=(f_size, f_size),
                                               sigma=deform_sigma) * f_size_float
        # correct for spherical distortion using 1/sin on x axis (horizontal)
        # assume the second dim is y, the third dim is x (indexing is ij)
        warp_y = warp[..., 0]
        warp_x = tf.math.divide_no_nan(warp[..., 1], S)
        warp = tf.stack([warp_y, warp_x], axis=-1)
        # integrate to get differomorphic deformation field
        fn = lambda x: vxm.utils.integrate_vec(x, nb_steps=7)
        warp = tf.map_fn(fn, warp)
        return warp

    if 'deform' in types:  # same random deformation is applied to all images in the list
        img_shape = tf.shape(img[0])
        warp_shape = [img_shape[0], img_shape[1], img_shape[2], 2]
        warp = random_deform(warp_shape)
        fn = lambda x: vxm.utils.transform(x[0], x[1], 'linear')
        img = [tf.map_fn(fn, (x, warp), fn_output_signature=tf.float32) for x in img]

    if not was_list:
        img = img[0]

    return img
