""" sandbox for generative models """

import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras.layers import Layer

import neurite as ne
import voxelmorph as vxm


# generator wrapper for single
def single_sample_gen(im_sample, input_shape, output_shape=None, input_init="zero"):
    """
    given a single image, keep returning the same image iteration:

    (input, (im_sample, input*0))

    input_shape is [batch_size, *vol_size, nb_labels]
    input_init can be:
        "zero"
        "rand"
        a volume of the size input_shape
    """
    if output_shape is None:
        output_shape = input_shape

    if isinstance(input_init, str) and input_init == "zero":
        inp = np.zeros(input_shape)
    elif isinstance(input_init, str) and input_init == "rand":
        inp = np.log(np.random.random(input_shape) + np.finfo(float).eps)

    else:
        # assumes passed in segmentation.
        inp = np.log(input_init + np.finfo(float).eps)

    while 1:
        yield (inp, [im_sample, np.zeros(output_shape)])


# define costs
def sse_cost(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true))


def prior_extract(Z, P, dim=3):
    """
    given "true" prior table P, and estimated prior Z, take the maximum prior Z
    along given dimension and compute P(argmax(Z))

    *warning* we approximate this to allow for derivatives, which might lead to some issues...
    """

    # a bit of a hack, but should comput P(L=l)
    return K.sum(P * ne.metrics._hard_max(Z, dim), -1)


class PriorCost():
    def __init__(self, prior_vol):
        self.prior = prior_vol
        self.log_prior = np.log(prior_vol + np.finfo(float).eps)

    def prior_cost(self, _, y_pred):
        return -K.sum(K.log(K.epsilon() + prior_extract(y_pred, self.prior)))


class LogPCost():
    def __init__(self, axis=3):
        self.axis = axis

    def log_max_p(self, _, y_pred):
        return -K.sum(K.log(K.epsilon() + K.max(y_pred, axis=self.axis)))

    def max_logp(self, _, y_pred):
        """ assumes y_pred is in logp already """
        return -K.sum(K.log(K.max(keras.activations.softmax(y_pred, axis=self.axis),
                                  axis=self.axis) + K.epsilon()))
        # return -K.sum(K.max(y_pred, axis=3))


def draw_shapes(
    shape, num_label=26, zero_max=0.5, fwhm_min=16, fwhm_max=64,
    warp_max=0.1, gen_scale=1, dtype=None, seed=None,
):
    '''
    Generate a label maps of N random shapes with labels {0, 1, ..., N-1}. Batch and feature
    dimensions are not supported.

    This is work in progress and should not be shared externally. If you would
    like to use the code for an abstract or paper, please contact the author.

    Author:
        mu40

    Arguments:
        shape: Spatial dimensions output label map.
        num_label: Number of random shapes N to generate.
        zero_max: Maximum proportion of shapes randomly set to zero.
        fwhm_min: Minimum blurring FWHM for shapes creation.
        fwhm_max: Maximum blurring FWHM for shapes creation.
        warp_max: Maximum warp SD as a proportion of the tensor size.
        gen_scale: Scale at which shapes are generated. A value of 2 means half
            resolution. Upsampling uses nearest-neighbor interpolation.
        dtype: Output data type.
        seed: Seed for reproducible randomization.

    Returns:
        Label map of random shapes.
    '''
    n_dim = len(shape)
    sub_shape = [s // gen_scale for s in shape]
    std_min = fwhm_min / gen_scale / 2.355
    std_max = fwhm_max / gen_scale / 2.355
    warp_max *= np.mean(shape)

    # Data types.
    compute_type = tf.keras.mixed_precision.global_policy().compute_dtype
    compute_type = tf.dtypes.as_dtype(compute_type)
    dtype = compute_type if dtype is None else tf.dtypes.as_dtype(dtype)

    # Randomization.
    rand = np.random.default_rng(seed)
    seed = lambda: rand.integers(np.iinfo(np.int32).max, dtype=np.int32)
    prop = lambda: dict(seed=seed(), dtype=compute_type)
    blur = lambda x: random_blur_rescale(x, std_min, std_max, rand=rand)

    # Images.
    v = tf.random.normal((num_label, *sub_shape, 1), **prop())
    v *= tf.random.uniform(shape=(num_label, *[1] * n_dim, 1), **prop())
    v = tf.map_fn(blur, v)

    # Transforms.
    t = tf.random.normal(shape=(num_label, *sub_shape, n_dim), **prop())
    t *= tf.random.uniform(shape=(), minval=1, maxval=warp_max, **prop())
    t *= tf.random.uniform(shape=(num_label, *[1] * n_dim, n_dim), **prop())
    t = tf.map_fn(blur, t)

    # Application and labels.
    perm = (*range(1, n_dim + 1), 0, n_dim + 1)
    v = tf.transpose(v, perm)
    t = tf.transpose(t, perm)
    x = vxm.utils.transform(v[..., 0], t)
    x = tf.argmax(x, axis=-1, output_type=tf.int32)[..., None]

    # Zero a fraction of labels in [0, maxval).
    if zero_max > 0:
        zero_max = min(np.int32(zero_max * num_label), num_label)
        n = tf.random.uniform(shape=(), maxval=zero_max, dtype=tf.int32, seed=seed())
        lut = tf.concat((
            tf.zeros(n, tf.float32),
            tf.range(1, num_label - n + 1, dtype=tf.float32),
        ), axis=0)
        x = tf.gather(lut, indices=x)

    x = ne.utils.zoom(x, zoom_factor=gen_scale, interp_method='nearest')
    return tf.cast(x, dtype) if x.dtype != dtype else x


def random_blur_rescale(
    x, std_min=8 / 2.355, std_max=32 / 2.355, isotropic=False,
    rand=None, reduce=tf.math.reduce_std, batched=False,
):
    warnings.warn('nes.utils.generative.random_blur_rescale is deprecated and will be removed in '
                  'the near future. Please use ne.utils.augment.random_blur_rescale instead.')

    seed = None if rand is None else rand.integers(np.iinfo(int).max)
    return ne.utils.augment.random_blur_rescale(x, std_min, std_max, isotropic, seed, reduce, batched)


def draw_perlin_full(
    shape, noise_min=0.01, noise_max=1,
    fwhm_min=4, fwhm_max=32, isotropic=False, batched=False,
    featured=False, reduce=tf.math.reduce_std, dtype=tf.float32, seed=None,
):
    warnings.warn('nes.utils.generative.draw_perlin_full is deprecated and will be removed in the '
                  'near future. Please use ne.utils.augment.draw_perlin_full instead.')
    return ne.utils.augment.draw_perlin_full(shape,
                                             noise_min,
                                             noise_max,
                                             fwhm_min,
                                             fwhm_max,
                                             isotropic,
                                             batched,
                                             featured,
                                             reduce,
                                             dtype,
                                             axes=None,
                                             seed=seed)


def sample_identical(shape, val_min=5, val_max=25, dtype=tf.float32, seed=None):
    '''
    Draw a random number from a uniform distribution and return a tensor of specified shape whose
    elements are all equal to the number drawn. For use with the labels-to-image model.

    Author:
        mu40
    '''
    return tf.ones(shape, dtype) * tf.random.uniform([], val_min, val_max, dtype, seed)


def sample_linked(
    shape, labels, linkages, mean_min=25, mean_max=225, dtype=tf.float32, seed=None,
):
    '''
    Draw linked label intensity means from uniform distributions, such that the
    means lie within a specified distance from specified reference labels. For use
    with the labels-to-image model: see example script.

    Author:
        mu40

    Parameters:
        shape: Output shape with dimensions: batch, channel, label.
        labels: List of all labels in the (training) data.
        linkages: List of label linkages, specifying the label of interest, the
            reference label to link to, and the maximum relative deviation from
            the target label as (LABEL, REF, MAX_DEV) for each label to replace.
            Pass 0.05 for a maximum deviation of 5%.
        mean_min: List of lower bounds on the original means drawn to generate the
            intensities for each label. If scalar, the value will be used for all labels.
        mean_min: List of upper bounds on the original means drawn to generate the
            intensities for each label. If scalar, the value will be used for all labels.
        dtype: Data type of the output.
        seed: Integer for reproducible randomization.

    Returns:
        Linked label-intensity samples as a tensor with dimensions: batch, channel, label.
    '''
    assert len(labels) == len(set(labels)), f'repeated labels in {labels}'
    ftype = tf.float32
    labels = sorted(set(labels))
    num_label = len(labels)

    # Sanity checks.
    list_old, list_new, list_diff = zip(*linkages)
    assert len(list_old) == len(set(list_old)), f'repeated source label in {list_old}'
    assert all(x not in list_old for x in list_new), 'label used both as source and target'
    assert all(x not in list_new for x in list_old), 'label used both as source and target'

    # Conversion.
    old_to_new = np.arange(num_label)
    diff_low = np.ones(num_label)
    diff_upp = np.ones(num_label)
    for old, new, diff in zip(list_old, list_new, list_diff):
        old = labels.index(old)
        new = labels.index(new)
        old_to_new[old] = new
        diff_low[old] = 1 - diff
        diff_upp[old] = 1 + diff

    # Reproducible randomization.
    rand = np.random.default_rng(seed)
    seed = lambda: rand.integers(np.iinfo(int).max)

    # Boundaries.
    bound = mean_min, mean_max
    bound = map(lambda x: tf.concat(x, axis=0), bound)
    bound = map(lambda x: x if x.dtype == ftype else tf.cast(x, ftype), bound)

    # Sampling.
    means_old = tf.random.uniform(shape, *bound, seed=seed(), dtype=ftype)
    means_new = tf.gather(means_old, indices=old_to_new, axis=-1)
    means_new *= tf.random.uniform(shape, diff_low, diff_upp, seed=seed(), dtype=ftype)

    if means_new.dtype != tf.dtypes.as_dtype(dtype):
        means_new = tf.cast(means_new, dtype)

    return means_new
