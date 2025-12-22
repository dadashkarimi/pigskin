"""
tensorflow/keras utilities for the neurite project

If you use this code, please cite 
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

# internal python import
import sys
import warnings

# third party
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL
from tensorflow.keras import backend as K

import neurite as ne
import voxelmorph as vxm
import voxelmorph_sandbox as vxms

# local imports
from . import layers
from . import utils


def EncoderNet(nb_features,
               input_shape,
               nb_levels,
               conv_size,
               name=None,
               prefix=None,
               feat_mult=1,
               pool_size=2,
               dilation_rate_mult=1,
               padding='same',
               activation='elu',
               layer_nb_feats=None,
               use_residuals=False,
               nb_conv_per_level=2,
               conv_dropout=0,
               dense_size=256,
               nb_labels=2,
               final_activation=None,
               rescale=None,
               dropout=None,
               batch_norm=None):
    """
    Fully Convolutional Encoder-based classifer
    if nb_labels is 0 assume it is a regression net and use linear activation
    (if None specified)
    """

    # allocate the encoder arm
    enc_model = conv_enc(nb_features,
                         input_shape,
                         nb_levels,
                         conv_size,
                         name=name,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         padding=padding,
                         activation=activation,
                         use_residuals=use_residuals,
                         nb_conv_per_level=nb_conv_per_level,
                         conv_dropout=conv_dropout,
                         batch_norm=batch_norm)

    # run the encoder outputs through a dense layer
    flat = KL.Flatten()(enc_model.outputs[0])
    if dropout is not None and dropout > 0:
        flat = KL.Dropout(dropout, name='dropout_flat')(flat)
    dense = KL.Dense(dense_size, name='dense')(flat)
    if dropout is not None and dropout > 0:
        dense = KL.Dropout(dropout, name='dropout_dense')(dense)

    if nb_labels <= 0:  # if labels <=0 assume a regression net
        nb_labels = 1
        if (final_activation is None):
            final_activation = 'linear'
    else:  # if labels>=1 assume a classification net
        if (final_activation is None):
            final_activation = 'softmax'

    if (rescale is not None):
        dense = layers.RescaleValues(rescale)(dense)
    out = KL.Dense(nb_labels, name='output_dense', activation=final_activation)(dense)
    model = keras.models.Model(inputs=enc_model.inputs, outputs=out)

    return model


def DenseLayerNet(inshape, layer_sizes,
                  nb_labels=2,
                  activation='relu',
                  final_activation='softmax',
                  dropout=None,
                  batch_norm=None):
    """
    A net made up of a set of dense layers connected to  a classification
    output. 
    if nb_labels is 0 assume it is a regression net and use linear activation
    (if None specified)
    """
    inputs = KL.Input(shape=inshape, name='input')
    prev_layer = KL.Flatten(name='flat_inputs')(inputs)

    # to prevent overfitting include some kernel and bias regularization
    kreg = keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)
    breg = keras.regularizers.l2(1e-4)

    # connect the list of dense layers to each other
    for lno, layer_size in enumerate(layer_sizes):
        prev_layer = KL.Dense(layer_size, name='dense%d' % lno, activation=activation,
                              kernel_regularizer=kreg, bias_regularizer=breg)(prev_layer)
        if dropout is not None:
            prev_layer = KL.Dropout(dropout, name='dropout%d' % lno)(prev_layer)
        if batch_norm is not None:
            prev_layer = KL.BatchNormalization(name='BatchNorm%d' % lno)(prev_layer)

    # tie the previous dense layer to a onehot encoded output layer
    last_layer = KL.Dense(nb_labels, name='last_dense', activation=final_activation)(prev_layer)

    model = keras.models.Model(inputs=inputs, outputs=last_layer)
    return (model)


###############################################################################
# Synthstrip networks
###############################################################################


class SynthStrip(ne.tf.modelio.LoadableModel):
    """
    SynthStrip model for learning subject-to-subject registration from images
    with arbitrary contrasts synthesized from label maps.

    Author: brf2
    """

    @ne.tf.modelio.store_config_args
    def __init__(self, inshape, labels_in, labels_out,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 src_feats=1,
                 gen_args={}):
        """
        Parameters:
            inshape: Input shape, e.g. (160, 160, 192).
            labels_in: List of all labels included in the training segmentations.
            labels_out: List of labels to encode in the output one-hot maps.
            gen_args: Keyword arguments passed to the internal generative model.
        """

        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        inshape = tuple(inshape)

        # take input label map and synthesize an image from it
        gen_model = labels_to_image(
            labels_in=labels_in,
            labels_out=labels_out,
            in_shape=inshape,
            id=0,
            return_def=False,
            one_hot=False,
            **gen_args
        )

        synth_image, synth_labels = gen_model.outputs

        # build a unet to apply to the synthetic image and strip it
        unet_model = ne.models.unet(
            nb_unet_features,
            inshape + (1,),
            nb_unet_levels,
            ndims * (3,),
            1,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            final_pred_activation='linear'
        )

        unet_outputs = unet_model(synth_image)

        # output the prob of brain and the warped label map
        # so that the loss function can compute brain/nonbrain
        stacked_output = KL.Concatenate(axis=-1, name='image_label')(
            [unet_outputs, tf.cast(synth_labels, tf.float32)])

        super().__init__(inputs=gen_model.inputs, outputs=stacked_output)

        # cache pointers to important layers and tensors for future reference
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.unet = unet_model     # this is the stripping net
        self.references.gen_model = gen_model
        self.references.synth_image = synth_image

    def get_strip_model(self):
        return self.references.unet


def labels_to_flash_image(
    labels_in,
    labels_out=None,
    in_shape=None,
    out_shape=None,
    input_model=None,
    num_chan=1,
    max_shift=0,
    max_rotate=0,
    max_scale=0,
    max_shear=0,
    normal_shift=False,   # Sample normally, treating max as SD.
    normal_rotate=False,  # Sample normally, treating max as SD.
    normal_scale=False,   # Sample normally, truncating beyond 2 SDs.
    normal_shear=False,   # Sample normally, treating max as SD.
    axes_flip=False,
    axes_swap=False,
    warp_res=16,
    warp_min=0,
    warp_max=1,
    warp_zero_mean=False,
    crop_min=0,
    crop_max=0.2,
    crop_prob=0,
    crop_axes=None,
    mean_min_T1=None,
    mean_max_T1=None,
    mean_min_PD=None,
    mean_max_PD=None,
    mean_func=None,
    noise_min=0.01,
    noise_max=0.15,
    zero_background=0.2,
    blur_min=0,
    blur_max=1,
    bias_res=40,
    bias_min=0,
    bias_max=0.3,
    bias_clip=None,
    bias_func=tf.exp,
    b1_plus_res=None,
    b1_plus_min=.2,
    b1_plus_max=.4,
    b1_plus_func=lambda x: tf.clip_by_value(x+1, .1, 5),
    b1_plus_clip=None,
    flip_range=[3*np.pi / 180, 60*np.pi / 180],
    TR_range=[5,80],
    slice_stride_min=3,
    slice_stride_max=6,
    slice_prob=0,
    slice_axes=None,
    clip_max=255,
    normalize=True,
    gamma=0.25,
    one_hot=True,
    half_res=False,
    seeds={},
    return_im=True,
    return_map=True,
    return_vel=False,
    return_def=False,
    return_aff=False,
    return_mean=False,
    return_image_no_bias=False,
    return_bias=False,
    compose_biases=False,
    return_b1_plus=False,
    return_b1_plus_ratio=False,
    return_composed_bias=False,
    return_PD=False,
    return_T1=False,
    return_mri_parms=False,
    id=0,
    image_interp='linear',
):
    '''Build model that augments label maps and synthesizes images from them.
       using a FLASH Bloch forward model
       Stolen shamelessly from labels_to_image, with which it shares all
       parameters except:

    return_b1_plus       - return the flip angle field
    return_b1_plus_ratio - return the ratio of the image with and without the 
                          nonuniform flip angle field
    b1_plus_res          - list of resolutions for the flip field
    b1_plus_min          - 1+min scaling for flip field
    b1_plus_max          - 1+max scaling for flip field
    b1_plus_func         - function to map the [mn,mx] to (0, inf).
    flip_range          - range of flip angles to sample from
    TR_range            - range of TRs to sample from


    gamma, normalize and clipping not implemted yet!!!
    '''

    # Compute type.
    major, minor, _ = map(int, tf.__version__.split('.'))
    if major > 2 or (major == 2 and minor > 3):
        compute_type = tf.keras.mixed_precision.global_policy().compute_dtype
        compute_type = tf.dtypes.as_dtype(compute_type)
    else:
        compute_type = tf.float32
    integer_type = tf.int32

    labels_input = KL.Input(shape=(*in_shape, 1), name=f'input_{id}', dtype=compute_type)

    # Dimensions.
    in_shape = np.asarray(labels_input.shape[1:-1])
    if out_shape is None:
        out_shape = in_shape
    out_shape = np.array(out_shape) // (2 if half_res else 1)
    num_dim = len(in_shape)
    batch_size = tf.shape(labels_input)[0]

    # Affine transform.
    def sample_motion(shape, lim, normal=False, truncate=False, seed=None, dtype=tf.float32):
        ''' Wrap TensorFlow function for random-number generation.'''
        prop = dict(shape=shape, seed=seed, dtype=dtype)
        if normal:
            func = 'truncated_normal' if truncate else 'normal'
            prop.update(dict(stddev=lim))
        else:
            func = 'uniform'
            prop.update(dict(minval=-lim, maxval=lim))
        return getattr(tf.random, func)(**prop)

    parameters = tf.concat((
        sample_motion(
            shape=(batch_size, num_dim), dtype=compute_type,
            lim=max_shift, normal=normal_shift, seed=seeds.get('shift'),
        ),
        sample_motion(
            shape=(batch_size, 1 if num_dim == 2 else 3), dtype=compute_type,
            lim=max_rotate, normal=normal_rotate, seed=seeds.get('rotate'),
        ),
        sample_motion(
            shape=(batch_size, num_dim), truncate=True, dtype=compute_type,
            lim=max_scale, normal=normal_scale, seed=seeds.get('scale'),
        ),
        sample_motion(
            shape=(batch_size, 1 if num_dim == 2 else 3), dtype=compute_type,
            lim=max_shear, normal=normal_shear, seed=seeds.get('shear'),
        ),
    ), axis=-1)

    affine = vxm.layers.ParamsToAffineMatrix(
        ndims=num_dim, deg=True, shift_scale=True, last_row=True,
    )(parameters)

    # Shift origin to rotate about image center.
    origin = np.eye(num_dim + 1)
    origin[:num_dim, -1] = -0.5 * (in_shape - 1)

    # Symmetric padding: center output volume within input volume.
    center = np.eye(num_dim + 1)
    center[:num_dim, -1] = np.round(0.5 * (in_shape - (2 if half_res else 1) * out_shape))

    # Apply transform in input space to avoid rescaling translations at half resolution.
    scale = np.diag((*[2 if half_res else 1] * num_dim, 1))
    trans = np.linalg.inv(origin) @ affine @ origin @ center @ scale

    # Convert to dense transform.
    trans = vxm.layers.AffineToDenseShift(out_shape, shift_center=False)(trans[:, :num_dim, :])

    # Diffeomorphic deformation.
    if warp_max > 0:
        # Velocity field.
        vel_draw = lambda x: ne.utils.augment.draw_perlin(
            out_shape=(*out_shape // (1 if half_res else 2), num_dim),
            scales=np.asarray(warp_res) / 2,
            min_std=warp_min, max_std=warp_max,
            dtype=compute_type, seed=seeds.get('warp'),
        )
        vel_field = KL.Lambda(lambda x: tf.map_fn(vel_draw, x), name=f'vel_{id}')(labels_input)
        if warp_zero_mean:
            vel_field -= tf.reduce_mean(vel_field, axis=range(1, num_dim + 1))
        def_field = vxm.layers.VecInt(int_steps=5, name=f'vec_int_{id}')(vel_field)
        if not half_res:
            def_field = vxm.layers.RescaleTransform(zoom_factor=2, name=f'def_{id}')(def_field)

        # Compose with affine.
        trans = vxm.layers.ComposeTransform(name=f'compose_{id}')([trans, def_field])

    # Apply transform.
    labels = vxm.layers.SpatialTransformer(
        interp_method='nearest', fill_value=0, name=f'trans_{id}',
    )([labels_input, trans])
    labels = tf.cast(labels, integer_type)

    # Generation labels default to all input labels.
    if not isinstance(labels_in, dict):
        labels_in = {lab: lab for lab in labels_in}
    # Lookup table rebasing generation labels into [0, 1, ..., N).
    labels_gen = np.unique(list(labels_in.values()))
    gen_to_ind = np.zeros(labels_gen[-1] + 1)
    for i, gen in enumerate(labels_gen):
        gen_to_ind[gen] = i
    # Rebase input labels.
    lut_in = np.zeros(max(labels_in) + 1, integer_type.as_numpy_dtype)
    for inp, gen in labels_in.items():
        lut_in[inp] = gen_to_ind[gen]
    indices = tf.gather(lut_in, indices=labels)

    # Intensity means and standard deviations.
    num_labels = len(labels_gen)
    if mean_min_PD is None:
        mean_min_PD = [1] + [100] * (num_labels - 1)
    if mean_max_PD is None:
        mean_max_PD = [1000] * num_labels
    mean_PD = tf.random.uniform(
        shape=(batch_size, 1, num_labels),    
        minval=mean_min_PD, maxval=mean_max_PD,
        dtype=compute_type, seed=seeds.get('mean_PD'),
    )
    if mean_min_T1 is None:
        mean_min_T1 = [1] + [100] * (num_labels - 1)
    if mean_max_T1 is None:
        mean_max_T1 = [1000] * num_labels
    mean_T1 = tf.random.uniform(
        shape=(batch_size, 1, num_labels),    
        minval=mean_min_T1, maxval=mean_max_T1,
        dtype=compute_type, seed=seeds.get('mean_T1'),
    )
    mean = tf.concat([mean_PD, mean_T1], axis=1)
    if mean_func is not None:
        mean = mean_func(
            shape=(batch_size, 2, num_labels),
            dtype=compute_type, seed=seeds.get('mean'),
        )

    # Label intensities.
    off_chan = tf.range(2) * num_labels
    off_batch = tf.range(batch_size) * 2 * num_labels
    indices += tf.reshape(off_batch, shape=(-1, *[1] * num_dim, 1)) + off_chan
    gather = lambda x: tf.gather(tf.reshape(x[0], shape=(-1,)), indices=x[1])
    mean = KL.Lambda(gather)([mean, indices])
    image = mean

    if b1_plus_res is not None:
        b1_plus_draw = lambda x: ne.utils.augment.draw_perlin(
            out_shape=(*out_shape, 1), min_std=b1_plus_min, max_std=b1_plus_max,
            scales=np.asarray(b1_plus_res) / (2 if half_res else 1),
            dtype=image.dtype, seed=seeds.get('b1_plus'),
        )
        b1_plus_field = KL.Lambda(lambda x: tf.map_fn(b1_plus_draw, x), name=f'b1_plus_draw_{id}')(image)
        b1_plus_field = KL.Lambda(b1_plus_func, name=f'b1_plus_func_{id}')(b1_plus_field)
        if b1_plus_clip is not None:
            print(f'clipping B1+ field to {b1_plus_clip}')
            clip_func = lambda x : tf.clip_by_value(x, b1_plus_clip[0], b1_plus_clip[1])
            b1_plus_field = KL.Lambda(clip_func, name=f'b1_plus_clip_{id}')(b1_plus_field)

        flip_angles = tf.random.uniform(
            shape=(batch_size, *[1], num_chan),
            dtype=compute_type, seed=seeds.get('flip_angles'),
        ) * (flip_range[1] - flip_range[0]) + flip_range[0]
        TRs = tf.random.uniform(
            shape=(batch_size, *[1], num_chan),
            dtype=compute_type, seed=seeds.get('TR'),
        ) * (TR_range[1] - TR_range[0]) + TR_range[0]
        E1 = tf.exp(-TRs / (image[..., 1:2]+.001))
        no_flip_field = tf.ones(tf.shape(b1_plus_field)) * flip_angles
        flip_field = b1_plus_field * flip_angles
        image_no_noise = image[..., 0:1] * tf.math.abs(tf.math.sin(flip_field)) * (1 - E1) / (1 - tf.math.abs(tf.math.cos(flip_field)) * E1)
        E1_mean = tf.exp(-TRs / (mean[..., 1:2]+.001))
        PD_mean = mean[..., 0:1]
        image_no_b1plus = PD_mean * tf.math.sin(no_flip_field) * (1 - E1_mean) / (1 - tf.math.cos(no_flip_field) * E1_mean)
        if return_b1_plus_ratio:
            b1_plus_ratio = tf.math.divide_no_nan(image_no_noise, image_no_b1plus, name='b1_plus_ratio')

    # Noise.
    image_no_bias = tf.clip_by_value(layers.GaussianNoise(noise_min, noise_max, seed=seeds.get('noise'))(
        image_no_noise), 0, 10000)

    # Bias field.
    if bias_max > 0:
        print(f'using bias_res list {bias_res}')
        bias_draw = lambda x: ne.utils.augment.draw_perlin(
            out_shape=(*out_shape, 1), min_std=bias_min, max_std=bias_max,
            scales=np.asarray(bias_res) / (2 if half_res else 1),
            dtype=image_no_bias.dtype, seed=seeds.get('bias'),
        )
        bias_field_before_clip = KL.Lambda(lambda x: tf.map_fn(bias_draw, x), name=f'bias_draw_{id}')(image_no_bias)
        bias_field = KL.Lambda(bias_func, name=f'bias_func_{id}')(bias_field_before_clip)

        if not compose_biases and bias_clip is not None:
            print(f'clipping B1- field to {bias_clip}')
            clip_func = lambda x : tf.clip_by_value(x, bias_clip[0], bias_clip[1])
            bias_field = KL.Lambda(clip_func, name=f'bias_clip_{id}')(bias_field)
        elif compose_biases:  # compose them then clip and recompute bias and image
            image_no_noise_no_bias = image_no_noise * bias_field
            bias_field = tf.math.divide_no_nan(image_no_noise_no_bias, image_no_noise)
            if bias_clip is not None:
                bias_field = tf.clip_by_value(bias_field, bias_clip[0], bias_clip[1])

        image = image_no_bias * bias_field
    else:
        image = image_no_bias

    # Blur.
    blur_min, blur_max = map(lambda x: [x] if np.isscalar(x) else x, (blur_min, blur_max))
    blur_min, blur_max = map(lambda x: np.ravel(x) / (2 if half_res else 1), (blur_min, blur_max))
    blur_min, blur_max = map(np.ndarray.tolist, (blur_min, blur_max))
    blur_min, blur_max = map(lambda x: x if len(x) > 1 else x * num_dim, (blur_min, blur_max))
    assert len(blur_min) == len(blur_max) == num_dim, 'unacceptable number of blurring sigmas'
    if any(x > 0 for x in blur_max):
        kernels = KL.Lambda(lambda x: ne.utils.gaussian_kernel(
            blur_max, min_sigma=blur_min, separate=True, random=blur_min != blur_max,
            dtype=x.dtype, seed=seeds.get('blur'),
        ))(image)
        image = ne.utils.separable_conv(image, kernels, batched=True)

    if labels_out is not None:
        if not isinstance(labels_out, dict):
            labels_out = {lab: lab for lab in labels_out}
        lut_out = np.zeros(max(labels_in) + 1, integer_type.as_numpy_dtype)
        for old, new in labels_out.items():
            if old in labels_in:
                lut_out[old] = new
    # For one-hot encoding, update the lookup table such that the M desired
    # output labels are rebased into the interval [0, M-1[. If the background
    # with value 0 is not part of the output labels, set it to -1 to remove it
    # from the one-hot maps.
    if one_hot:
        labels_hot = np.unique(list(labels_out.values() if labels_out else labels_in))
        lut_hot = np.full(labels_hot[-1] + 1, fill_value=-1, dtype=integer_type.as_numpy_dtype)
        for i, lab in enumerate(labels_hot):
            lut_hot[lab] = i
        lut_out = lut_hot[lut_out] if labels_out else lut_hot

    # Convert input to output labels only once.
    if labels_out or one_hot:
        labels = tf.gather(lut_out, indices=labels, name=f'labels_back_{id}')
    if one_hot:
        labels = tf.one_hot(labels[..., 0], depth=len(labels_hot), dtype=compute_type)

    outputs = []

    if return_im:
        outputs.append(image)
    if return_map:
        outputs.append(labels)
    if return_def:
        outputs.append(def_field)
    if return_aff:
        outputs.append(affine)
    if return_image_no_bias:
        outputs.append(image_no_bias)    
    if return_mean:
        outputs.append(mean)
    if return_bias:
        outputs.append(bias_field)
    if return_b1_plus:
        outputs.append(b1_plus_field)
    if return_b1_plus_ratio:
        outputs.append(b1_plus_ratio)
    if return_PD:
        outputs.append(mean_PD)
    if return_T1:
        outputs.append(mean_T1)
    if return_mri_parms:
        outputs.append(tf.concat([TRs, flip_angles], axis=1))
    model = tf.keras.Model(labels_input, outputs)
    return model


def labels_to_image(
    labels_in,
    labels_out=None,
    in_shape=None,
    out_shape=None,
    input_model=None,
    num_chan=1,
    aff_shift=0,
    aff_rotate=0,
    aff_scale=0,
    aff_shear=0,
    aff_normal_shift=False,
    aff_normal_rotate=False,
    aff_normal_scale=False,
    aff_normal_shear=False,
    axes_flip=False,
    axes_swap=False,
    warp_min=0.01,
    warp_max=1,
    warp_blur_min=(8, 8),
    warp_blur_max=(32, 32),
    warp_zero_mean=False,
    crop_min=0,
    crop_max=0.2,
    crop_prob=0,
    crop_axes=None,
    mean_min=None,
    mean_max=None,
    mean_func=None,
    noise_min=0.1,
    noise_max=0.2,
    zero_background=0,
    motion_steps=0,
    motion_max_shift=5,
    motion_max_rotate=5,
    motion_shuffle_lines=True,
    blur_min=0,
    blur_max=1,
    bias_min=0.01,
    bias_max=0.1,
    bias_blur_min=32,
    bias_blur_max=64,
    bias_func=tf.exp,
    slice_stride_min=3,
    slice_stride_max=6,
    slice_prob=0,
    slice_axes=None,
    clip_max=255,
    normalize=True,
    gamma=0.5,
    one_hot=True,
    half_res=False,
    seeds={},
    return_im=True,
    return_map=True,
    return_vel=False,
    return_def=False,
    return_aff=False,
    return_mean=False,
    return_bias=False,
    id=0,
    **kwargs,
):
    """Build model that augments label maps and synthesizes images from them.

    This is work in progress and should not be shared externally.

    Parameters:
        labels_in: All possible input label values as an iterable. Passing a dictionary will remap
            input labels to the generation labels defined by the dictionary values, for example to
            draw the same intensity for left and right cortex. Generation labels must be hashable
            but not necessarily numeric. Remapping has no effect on the output labels.
        labels_out: Subset of the input labels to include in the output label maps, as an iterable.
            Passing a dictionary will remap the included input labels, for example to GM, WM, and
            CSF. Any input label missing in `labels_out` will be set to 0 (background) or excluded
            if one-hot encoding. None means the output labels will be the same as the input labels.
            Output labels must be numeric.
        in_shape: Spatial dimensions of the input label maps as an iterable.
        out_shape: Spatial dimensions of the outputs as an iterable. Inputs will be symmetrically
            cropped or zero-padded to fit. None means `in_shape`.
        input_model: Model whose outputs will be used as data inputs, and whose inputs will be used
            as inputs to the returned model.
        num_chan: Number of image channels to synthesize.
        aff_shift: Upper bound on the magnitude of translations, drawn uniformly in voxels.
        aff_rotate: Upper bound on the magnitude of rotations, drawn uniformly in degrees.
        aff_scale: Upper bound on the difference of the scaling magnitude from 1, drawn uniformly.
        aff_shear: Upper bound on the shearing magnitude, drawn uniformly.
        aff_normal_shift: Sample translations normally, with `aff_shift` as SD.
        aff_normal_rotate: Sample rotation angles normally, with `aff_rotate` as SD.
        aff_normal_scale: Sample scaling normally, with `aff_scale` as SD, truncating beyond 2 SDs.
        aff_normal_shear: Sample shearing normally, with `aff_shear` as SD.
        axes_flip: Randomly flip axes of the label map before generating the image. Should not be
            used with lateralized labels.
        axes_swap: Randomly permute axes of the label map before generating the image. Should not
            be used with lateralized labels. Requires an isotropic output shape.
        warp_min: Lower bound on the SDs used when drawing the SVF.
        warp_max: Upper bound on the SDs used when drawing the SVF.
        warp_zero_mean: Ensure that the SVF components have zero mean.
        crop_min: Lower bound on the proportion of the FOV to crop.
        crop_max: Upper bound on the proportion of the FOV to crop.
        crop_prob: Probability that we crop the FOV along an axis.
        crop_axes: Axes from which to draw for FOV cropping. None means all spatial axes.
        mean_min: List of lower bounds on the intensities drawn for each label. None means 0.
        mean_max: List of upper bounds on the intensities drawn for each label. None means 1.
        mean_func: Optional function called for drawing intensity means. Wrap e.g.
            nes.utils.generative.sample_linked with your own function defining
            rules for your specific training label (see script with example).
        noise_min: Lower bound on the noise SD relative to the max image intensity, 0.1 means 10%.
        noise_max: Upper bound on the noise SD relative to the max image intensity, 0.1 means 10%.
        zero_background: Probability that we set the background to zero.
        blur_min: Lower bound on the smoothing SDs. Can be a scalar or list.
        blur_max: Upper bound on the smoothing SDs. Can be a scalar or list.
        bias_min: Lower bound on the bias sampling SDs.
        bias_max: Upper bound on the bias sampling SDs.
        bias_blur_min: Lower bound on the bias smoothing FWHM.
        bias_blur_max: Upper bound on the bias smoothing FWHM.
        bias_func: Function applied voxel-wise to condition the bias field.
        slice_stride_min: Lower bound on slice thickness in original voxel units.
        slice_stride_max: Upper bound on slice thickness in original voxel units.
        slice_prob: Probability that we subsample to create thick slices.
        slice_axes: Axes from which to draw slice normal direction. None means all spatial axes.
        clip_max: Maximum value to clip to. Pass a list to sample from an interval.
        normalize: Min-max normalize the image.
        gamma: Maximum deviation of the gamma augmentation exponent from 1. Careful: without
            normalization, you may attempt to compute random exponents of negative numbers.
        one_hot: One-hot encode the output label map.
        seeds: Seeds for reproducible randomization or synchronization of components of this model
            across multiple instances. Pass a dictionary linking specific component names to integer
            seeds. If you pass an iterable of component names, seeds will be auto-generated. You
            will have to check the source code to identify component names.
        return_vel: Return the half-resolution SVF.
        return_def: Return the combined displacement field.
        return_aff: Return the (N+1)-by-(N+1) affine transformation matrix.
        return_mean: Return an uncorrupted copy of the image.
        return_bias: Return the applied bias field.
        id: Model identifier used to create unique layer names.

    Returns:
        Label-deformation and image-synthesis model.

    Author:
        mu40

    If you find this model useful, please cite:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879

        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    # Parameter name changes.
    def warn_pop(old, new):
        warnings.warn(f'Argument `{old}` to nes.models.labels_to_image has '
                      f'been deprecated in favor of `{new}`.')
        return kwargs.pop(old)

    if 'max_shift' in kwargs:
        aff_shift = warn_pop(old='max_shift', new='aff_shift')

    if 'max_rotate' in kwargs:
        aff_rotate = warn_pop(old='max_rotate', new='aff_rotate')

    if 'max_scale' in kwargs:
        aff_scale = warn_pop(old='max_scale', new='aff_scale')

    if 'max_shear' in kwargs:
        aff_shear = warn_pop(old='max_shear', new='aff_shear')

    if 'normal_shift' in kwargs:
        aff_normal_shift = warn_pop(old='normal_shift', new='aff_normal_shift')

    if 'normal_rotate' in kwargs:
        aff_normal_rotate = warn_pop(old='normal_rotate', new='aff_normal_rotate')

    if 'normal_scale' in kwargs:
        aff_normal_scale = warn_pop(old='normal_scale', new='aff_normal_scale')

    if 'normal_shear' in kwargs:
        aff_normal_shear = warn_pop(old='normal_shear', new='aff_normal_shear')

    # Default value changes.
    def warn_def(par, val):
        warnings.warn(f'In the future, the default value of argument `{par}` '
                      f'to nes.models.labels_to_image will change to {val}.')

    if clip_max == 255:
        warn_def(par='clip_max', val='None')

    if warp_max == 1:
        warn_def(par='warp_max', val='2')

    if slice_stride_min == 3:
        warn_def(par='slice_stride_min', val='1')

    if slice_stride_max == 6:
        warn_def(par='slice_stride_max', val='8')

    # Operation order changes.
    bias_first = kwargs.pop('bias_first', False)
    if not bias_first:
        warnings.warn('In the future, nes.models.labels_to_image will apply '
                      'the bias field just before adding noise. Use the new '
                      'code by passing `bias_first=True`.')

    assert not kwargs, f'unknown argument in {kwargs}'

    # Compute type.
    compute_type = tf.keras.mixed_precision.global_policy().compute_dtype
    compute_type = tf.dtypes.as_dtype(compute_type)
    integer_type = tf.int32

    # Seeds.
    if isinstance(seeds, str):
        seeds = [seeds]
    if not isinstance(seeds, dict):
        seeds = {f: hash(f) for f in seeds}

    # Input model.
    if input_model is None:
        labels = KL.Input(shape=(*in_shape, 1), name=f'input_{id}', dtype=compute_type)
        input_model = tf.keras.Model(*[labels] * 2)
    labels = input_model.output
    if labels.dtype != compute_type:
        labels = tf.cast(labels, compute_type)

    # Dimensions.
    in_shape = np.asarray(labels.shape[1:-1])
    if out_shape is None:
        out_shape = in_shape
    out_shape = np.array(out_shape) // (2 if half_res else 1)
    num_dim = len(in_shape)
    batch_size = tf.shape(labels)[0]

    # Affine transform.
    parameters = vxm.layers.DrawAffineParams(
        shift=aff_shift,
        rot=aff_rotate,
        scale=aff_scale,
        shear=aff_shear,
        normal_shift=aff_normal_shift,
        normal_rot=aff_normal_rotate,
        normal_scale=aff_normal_scale,
        normal_shear=aff_normal_shear,
        ndims=num_dim,
        dtype=compute_type,
        seeds={t: seeds.pop(t, None) for t in ('shift', 'rot', 'scale', 'shear')},
    )(labels)
    affine = vxm.layers.ParamsToAffineMatrix(
        ndims=num_dim, deg=True, shift_scale=True, last_row=True,
    )(parameters)

    # Shift origin to rotate about image center.
    origin = np.eye(num_dim + 1)
    origin[:num_dim, -1] = -0.5 * (in_shape - 1)

    # Symmetric padding: center output volume within input volume.
    center = np.eye(num_dim + 1)
    center[:num_dim, -1] = np.round(0.5 * (in_shape - (2 if half_res else 1) * out_shape))

    # Apply transform in input space to avoid rescaling translations at half resolution.
    scale = np.diag((*[2 if half_res else 1] * num_dim, 1))
    trans = np.linalg.inv(origin) @ affine @ origin @ center @ scale

    # Axis randomization.
    if axes_flip:
        trans = KL.Lambda(lambda x: x @ vxm.utils.draw_flip_matrix(
            out_shape, shift_center=False, dtype=compute_type, seed=seeds.pop('flip', None),
        ))(trans)
    if axes_swap:
        assert all(x == out_shape[0] for x in out_shape), 'non-isotropic output shape'
        trans = KL.Lambda(lambda x: x @ vxm.utils.draw_swap_matrix(
            num_dim, dtype=compute_type, seed=seeds.pop('swap', None),
        ))(trans)

    # Convert to dense transform.
    trans = vxm.layers.AffineToDenseShift(out_shape, shift_center=False)(trans[:, :num_dim, :])

    # Diffeomorphic deformation.
    if warp_max > 0:
        vel_field = ne.layers.PerlinNoise(
            shape=(*out_shape // (1 if half_res else 2), num_dim),
            noise_min=warp_min,
            noise_max=warp_max,
            isotropic=False,
            fwhm_min=np.asarray(warp_blur_min) / 2,
            fwhm_max=np.asarray(warp_blur_max) / 2,
            reduce=tf.math.reduce_max,
            axes=-1,
            dtype=compute_type,
            seed=seeds.pop('warp', None),
        )(labels)
        if warp_zero_mean:
            vel_field -= tf.reduce_mean(vel_field, axis=range(1, num_dim + 1))
        def_field = vxm.layers.VecInt(int_steps=5, name=f'vec_int_{id}')(vel_field)
        if not half_res:
            def_field = vxm.layers.RescaleTransform(zoom_factor=2, name=f'def_{id}')(def_field)

        # Compose with affine.
        trans = vxm.layers.ComposeTransform()([trans, def_field])

    # Apply transform.
    labels = vxm.layers.SpatialTransformer(
        interp_method='nearest', fill_value=0, name=f'trans_{id}',
    )([labels, trans])
    labels = tf.cast(labels, integer_type)

    # Cropping.
    labels = ne.layers.RandomCrop(
        crop_min=crop_min,
        crop_max=crop_max,
        prob=crop_prob,
        axis=crop_axes,
        seed=seeds.pop('crop', None),
    )(labels)

    # Generation labels. Default to all input labels.
    if not isinstance(labels_in, dict):
        labels_in = {i: i for i in labels_in}
    labels_gen = set(labels_in.values())

    # Rebase into [0, N): LUT from input to generation label to index.
    ind = {gen: i for i, gen in enumerate(labels_gen)}
    lut = [ind.get(labels_in.get(i), 0) for i in range(max(labels_in) + 1)]
    lut = tf.cast(lut, integer_type)
    indices = tf.gather(lut, indices=labels)

    # Intensity means and standard deviations.
    num_label = len(labels_gen)
    if mean_min is None:
        mean_min = [0] * num_label
    if mean_max is None:
        mean_max = [1] * num_label
    if mean_func is None:
        mean = tf.random.uniform(
            shape=(batch_size, num_chan, num_label),
            minval=mean_min,
            maxval=mean_max,
            dtype=compute_type,
            seed=seeds.pop('mean', None),
        )
    else:
        mean = mean_func(
            shape=(batch_size, num_chan, num_label),
            dtype=compute_type,
            seed=seeds.get('mean', None),
        )

    # Label intensities.
    off_chan = tf.range(num_chan) * num_label
    off_batch = tf.range(batch_size) * num_chan * num_label
    indices += tf.reshape(off_batch, shape=(-1, *[1] * num_dim, 1)) + off_chan
    mean = tf.gather(tf.reshape(mean, shape=(-1,)), indices=indices)
    image = mean

    # Bias field.
    if bias_first and bias_max > 0:
        bias_field = ne.layers.PerlinNoise(
            noise_min=bias_min,
            noise_max=bias_max,
            isotropic=False,
            fwhm_min=bias_blur_min / (2 if half_res else 1),
            fwhm_max=bias_blur_max / (2 if half_res else 1),
            reduce=tf.math.reduce_max,
            dtype=compute_type,
            seed=seeds.pop('bias', None),
        )(image)
        bias_field = KL.Lambda(bias_func, name=f'bias_func_{id}')(bias_field)
        image *= bias_field

    # Noise.
    image = ne.layers.GaussianNoise(noise_min, noise_max, seed=seeds.pop('noise', None))(image)

    # Motion in k-space.
    if motion_steps > 0:
        image = vxms.layers.RigidMotionSynth(
            max_shift=motion_max_shift, max_rotate=motion_max_rotate,
            nb_motion=motion_steps, shuffle_lines=motion_shuffle_lines, name=f'rigid_motion_{id}',
        )(image)

    # Background clearing.
    if zero_background > 0:
        bg_rand = tf.random.uniform(
            shape=(batch_size, *[1] * num_dim, 1),
            dtype=compute_type,
            seed=seeds.pop('background', None),
        )
        bg_zero = tf.math.less(bg_rand, zero_background)
        bg_zero = tf.math.logical_and(tf.equal(labels, 0), bg_zero)
        bg_zero = tf.math.logical_xor(True, bg_zero)
        image *= tf.cast(bg_zero, compute_type)

    # Blur.
    image = ne.layers.GaussianBlur(
        sigma=blur_max,
        min_sigma=blur_min,
        random=True,
        seed=seeds.pop('blur', None),
    )(image)

    # Bias field.
    if not bias_first and bias_max > 0:
        bias_field = ne.layers.PerlinNoise(
            noise_min=bias_min,
            noise_max=bias_max,
            isotropic=False,
            fwhm_min=bias_blur_min / (2 if half_res else 1),
            fwhm_max=bias_blur_max / (2 if half_res else 1),
            reduce=tf.math.reduce_max,
            dtype=compute_type,
            seed=seeds.pop('bias', None),
        )(image)
        bias_field = KL.Lambda(bias_func, name=f'bias_func_{id}')(bias_field)
        image *= bias_field

    # Create thick slices.
    image = ne.layers.Subsample(
        prob=slice_prob,
        stride_min=max(1, slice_stride_min / (2 if half_res else 1)),
        stride_max=max(1, slice_stride_max / (2 if half_res else 1)),
        axes=slice_axes,
        seed=seeds.pop('slice', None),
    )(image)

    # Intensity manipulations.
    image = ne.layers.RandomClip(
        clip_min=0,
        clip_max=clip_max,
        prob_min=1 if clip_max else 0,
        prob_max=1 if clip_max else 0,
        axes=(0, -1),
    )(image)
    if normalize:
        image = KL.Lambda(lambda x: tf.map_fn(ne.utils.minmax_norm, x))(image)
    if gamma > 0:
        assert 0 < gamma < 1, f'gamma value {gamma} outside interval [0, 1)'
        gamma = tf.random.uniform(
            shape=(batch_size, *[1] * num_dim, num_chan),
            minval=1 - gamma,
            maxval=1 + gamma,
            dtype=image.dtype,
            seed=seeds.pop('gamma', None),
        )
        image = tf.pow(image, gamma)

    # Output labels. Recode the original input labels if desired.
    lut = list(labels_in) if labels_out is None else labels_out
    if not isinstance(lut, dict):
        lut = {i: i for i in lut}
    labels_out = set(lut.values())

    # Rebase the M desired output labels into [0, M): LUT from input to index.
    if one_hot:
        ind = {out: i for i, out in enumerate(labels_out)}
        lut = {inp: ind[out] for inp, out in lut.items()}

    # Conversion. Set labels to -1 to exclude them from the one-hot maps.
    if any(k != lut[k] for k in lut) or set(labels_in) - set(lut):
        lut = [lut.get(i, -1 if one_hot else 0) for i in range(max(labels_in) + 1)]
        lut = tf.cast(lut, integer_type)
        labels = tf.gather(lut, indices=labels)

    if one_hot:
        labels = tf.one_hot(labels[..., 0], depth=len(labels_out), dtype=compute_type)

    outputs = []
    if return_im:
        outputs.append(image)
    if return_map:
        outputs.append(labels)
    if return_vel:
        outputs.append(vel_field)
    if return_def:
        outputs.append(def_field)
    if return_aff:
        outputs.append(affine)
    if return_mean:
        outputs.append(mean)
    if return_bias:
        outputs.append(bias_field)

    assert not seeds, f'unknown seeds {seeds}'
    return tf.keras.Model(input_model.inputs, outputs)


def image_to_image(
    in_shape=None,
    out_shape=None,
    num_chan=1,
    input_model=None,
    aff_shift=0,
    aff_rotate=0,
    aff_scale=0,
    aff_shear=0,
    aff_normal_shift=False,
    aff_normal_rotate=False,
    aff_normal_scale=False,
    aff_normal_shear=False,
    axes_flip=False,
    axes_swap=False,
    warp_min=0.01,
    warp_max=1,
    warp_blur_min=(8, 8),
    warp_blur_max=(32, 32),
    warp_zero_mean=False,
    crop_min=0,
    crop_max=0.2,
    crop_prob=0,
    crop_axes=None,
    lut_blur_min=32,
    lut_blur_max=64,
    noise_min=0.01,
    noise_max=0.10,
    blur_min=0,
    blur_max=1,
    bias_min=0.01,
    bias_max=0.1,
    bias_blur_min=32,
    bias_blur_max=64,
    bias_func=tf.exp,
    slice_stride_min=3,
    slice_stride_max=6,
    slice_prob=0,
    slice_axes=None,
    clip_max=255,
    normalize=True,
    gamma=0.5,
    half_res=False,
    seeds={},
    return_vel=False,
    return_def=False,
    return_aff=False,
    return_composed=False,
    image_interp='linear',
    id=0,
    **kwargs,
):
    """Build model for image augmentation.

    This is work in progress and should not be shared externally.

    Parameters:
        in_shape: Spatial dimensions of the input label maps as an iterable.
        out_shape: Spatial dimensions of the outputs as an iterable. Inputs will be symmetrically
            cropped or zero-padded to fit. None means `in_shape`.
        num_chan: Number of input image channels.
        input_model: Model whose outputs will be used as data inputs, and whose inputs will be used
            as inputs to the returned model.
        aff_shift: Upper bound on the magnitude of translations, drawn uniformly in voxels.
        aff_rotate: Upper bound on the magnitude of rotations, drawn uniformly in degrees.
        aff_scale: Upper bound on the difference of the scaling magnitude from 1, drawn uniformly.
        aff_shear: Upper bound on the shearing magnitude, drawn uniformly.
        aff_normal_shift: Sample translations normally, with `aff_shift` as SD.
        aff_normal_rotate: Sample rotation angles normally, with `aff_rotate` as SD.
        aff_normal_scale: Sample scaling normally, with `aff_scale` as SD, truncating beyond 2 SDs.
        aff_normal_shear: Sample shearing normally, with `aff_shear` as SD.
        axes_flip: Randomly flip axes of the image.
        axes_swap: Randomly permute axes of the image. Requires an isotropic
            output shape.
        warp_min: Lower bound on the SDs used when drawing the SVF.
        warp_max: Upper bound on the SDs used when drawing the SVF.
        warp_zero_mean: Ensure that the SVF components have zero mean.
        crop_min: Lower bound on the proportion of the FOV to crop.
        crop_max: Upper bound on the proportion of the FOV to crop.
        crop_prob: Probability that we crop the FOV along an axis.
        crop_axes: Axes from which to draw for FOV cropping. None means all spatial axes.
        lut_blur_min: Lower bound on the smoothing SD for random-contrast lookup.
        lut_blur_max: Upper bound on the smoothing SD for random-contrast lookup. Disabled if zero.
        noise_min: Lower bound on the noise SD relative to the max image intensity, 0.1 means 10%.
        noise_max: Upper bound on the noise SD relative to the max image intensity, 0.1 means 10%.
        blur_min: Lower bound on the smoothing SDs. Can be a scalar or list.
        blur_max: Upper bound on the smoothing SDs. Can be a scalar or list.
        bias_min: Lower bound on the bias sampling SDs.
        bias_max: Upper bound on the bias sampling SDs.
        bias_blur_min: Lower bound on the bias smoothing FWHM.
        bias_blur_max: Upper bound on the bias smoothing FWHM.
        bias_func: Function applied voxel-wise to condition the bias field.
        slice_stride_min: Lower bound on slice thickness in original voxel units.
        slice_stride_max: Upper bound on slice thickness in original voxel units.
        slice_prob: Probability that we subsample to create thick slices.
        slice_axes: Axes from which to draw slice normal direction. None means all spatial axes.
        clip_max: Maximum value to clip to. Pass a list to sample from an interval.
        normalize: Min-max normalize the image.
        gamma: Maximum deviation of the gamma augmentation exponent from 1. Careful: without
            normalization, you may attempt to compute random exponents of negative numbers.
        seeds: Seeds for reproducible randomization or synchronization of components of this model
            across multiple instances. Pass a dictionary linking specific component names to integer
            seeds. If you pass an iterable of component names, seeds will be auto-generated. You
            will have to check the source code to identify component names.
        return_vel: Return the half-resolution SVF.
        return_def: Return the combined displacement field.
        return_aff: Return the (N+1)-by-(N+1) affine transformation matrix.
        return_composed: Return the composed warp and affine transform.
        image_interp: Interpolation method for image spatial transform.
        id: Model identifier used to create unique layer names.

    Returns:
        Image-augmentation model.

    Author:
        mu40

    If you find this model useful, please cite:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879

        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    # Parameter name changes.
    def warn_pop(old, new):
        warnings.warn(f'Argument `{old}` to nes.models.image_to_image has '
                      f'been deprecated in favor of `{new}`.')
        return kwargs.pop(old)

    if 'max_shift' in kwargs:
        aff_shift = warn_pop(old='max_shift', new='aff_shift')

    if 'max_rotate' in kwargs:
        aff_rotate = warn_pop(old='max_rotate', new='aff_rotate')

    if 'max_scale' in kwargs:
        aff_scale = warn_pop(old='max_scale', new='aff_scale')

    if 'max_shear' in kwargs:
        aff_shear = warn_pop(old='max_shear', new='aff_shear')

    if 'normal_shift' in kwargs:
        aff_normal_shift = warn_pop(old='normal_shift', new='aff_normal_shift')

    if 'normal_rotate' in kwargs:
        aff_normal_rotate = warn_pop(old='normal_rotate', new='aff_normal_rotate')

    if 'normal_scale' in kwargs:
        aff_normal_scale = warn_pop(old='normal_scale', new='aff_normal_scale')

    if 'normal_shear' in kwargs:
        aff_normal_shear = warn_pop(old='normal_shear', new='aff_normal_shear')

    # Default value changes.
    def warn_def(par, val):
        warnings.warn(f'In the future, the default value of argument `{par}` '
                      f'to nes.models.image_to_image will change to {val}.')

    if clip_max == 255:
        warn_def(par='clip_max', val='None')

    if warp_max == 1:
        warn_def(par='warp_max', val='2')

    if slice_stride_min == 3:
        warn_def(par='slice_stride_min', val='1')

    if slice_stride_max == 6:
        warn_def(par='slice_stride_max', val='8')

    # Operation order changes.
    bias_first = kwargs.pop('bias_first', False)
    if not bias_first:
        warnings.warn('In the future, nes.models.image_to_image will apply '
                      'the bias field just before adding noise. Use the new '
                      'code by passing `bias_first=True`.')

    assert not kwargs, f'unknown argument in {kwargs}'

    # Compute type.
    compute_type = tf.keras.mixed_precision.global_policy().compute_dtype
    compute_type = tf.dtypes.as_dtype(compute_type)
    integer_type = tf.int32

    # Seeds.
    if isinstance(seeds, str):
        seeds = [seeds]
    if not isinstance(seeds, dict):
        seeds = {f: hash(f) for f in seeds}

    # Input model.
    if input_model is None:
        image = KL.Input(shape=(*in_shape, num_chan), name=f'input_{id}', dtype=compute_type)
        input_model = tf.keras.Model(*[image] * 2)
    image = input_model.output
    if image.dtype != compute_type:
        image = tf.cast(image, compute_type)

    # Dimensions.
    in_shape = np.asarray(image.shape[1:-1])
    if out_shape is None:
        out_shape = in_shape
    out_shape = np.array(out_shape) // (2 if half_res else 1)
    num_dim = len(in_shape)
    batch_size = tf.shape(image)[0]
    num_chan = image.shape[-1]

    # Affine transform.
    parameters = vxm.layers.DrawAffineParams(
        shift=aff_shift,
        rot=aff_rotate,
        scale=aff_scale,
        shear=aff_shear,
        normal_shift=aff_normal_shift,
        normal_rot=aff_normal_rotate,
        normal_scale=aff_normal_scale,
        normal_shear=aff_normal_shear,
        ndims=num_dim,
        dtype=compute_type,
        seeds={t: seeds.pop(t, None) for t in ('shift', 'rot', 'scale', 'shear')},
    )(image)
    affine = vxm.layers.ParamsToAffineMatrix(
        ndims=num_dim, deg=True, shift_scale=True, last_row=True,
    )(parameters)

    # Shift origin to rotate about image center.
    origin = np.eye(num_dim + 1)
    origin[:num_dim, -1] = -0.5 * (in_shape - 1)

    # Symmetric padding: center output volume within input volume.
    center = np.eye(num_dim + 1)
    center[:num_dim, -1] = np.round(0.5 * (in_shape - (2 if half_res else 1) * out_shape))

    # Apply transform in input space to avoid rescaling translations at half resolution.
    scale = np.diag((*[2 if half_res else 1] * num_dim, 1))
    trans = np.linalg.inv(origin) @ affine @ origin @ center @ scale

    # Axis randomization.
    if axes_flip:
        trans = KL.Lambda(lambda x: x @ vxm.utils.draw_flip_matrix(
            out_shape, shift_center=False, dtype=compute_type, seed=seeds.pop('flip', None),
        ))(trans)
    if axes_swap:
        assert all(x == out_shape[0] for x in out_shape), 'non-isotropic output shape'
        trans = KL.Lambda(lambda x: x @ vxm.utils.draw_swap_matrix(
            num_dim, dtype=compute_type, seed=seeds.pop('swap', None),
        ))(trans)

    # Convert to dense transform.
    trans = vxm.layers.AffineToDenseShift(out_shape, shift_center=False)(trans[:, :num_dim, :])

    # Diffeomorphic deformation.
    if warp_max > 0:
        vel_field = ne.layers.PerlinNoise(
            shape=(*out_shape // (1 if half_res else 2), num_dim),
            noise_min=warp_min,
            noise_max=warp_max,
            isotropic=False,
            fwhm_min=np.asarray(warp_blur_min) / 2,
            fwhm_max=np.asarray(warp_blur_max) / 2,
            reduce=tf.math.reduce_max,
            axes=-1,
            dtype=compute_type,
            seed=seeds.pop('warp', None),
        )(image)
        if warp_zero_mean:
            vel_field -= tf.reduce_mean(vel_field, axis=range(1, num_dim + 1))
        def_field = vxm.layers.VecInt(int_steps=5, name=f'vec_int_{id}')(vel_field)
        if not half_res:
            def_field = vxm.layers.RescaleTransform(zoom_factor=2, name=f'def_{id}')(def_field)

        # Compose with affine.
        trans = vxm.layers.ComposeTransform()([trans, def_field])

    # Apply transform.
    image = vxm.layers.SpatialTransformer(
        interp_method=image_interp, fill_value=0, name=f'trans_{id}',
    )([image, trans])

    # Cropping.
    image = ne.layers.RandomCrop(
        crop_min=crop_min,
        crop_max=crop_max,
        prob=crop_prob,
        axis=crop_axes,
        seed=seeds.pop('crop', None),
    )(image)

    # Random contrast lookup.
    if lut_blur_max > 0:
        lut_num = 256
        lut_max = lut_num - 1
        lut_draw = 1024

        # Smooth table. Filter shape: space, in, out.
        kernel = KL.Lambda(lambda x: ne.utils.gaussian_kernel(
            lut_blur_max,
            min_sigma=lut_blur_min,
            random=lut_blur_min != lut_blur_max,
            dtype=x.dtype,
            seed=seeds.get('lut'),
        ))(image)
        kernel = tf.reshape(kernel, shape=(-1, 1, 1))
        lut = tf.random.uniform(
            shape=(batch_size, lut_draw, 1),
            minval=0,
            maxval=lut_max,
            dtype=compute_type,
            seed=seeds.pop('lut', None),
        )
        lut = tf.nn.convolution(lut, kernel, padding='SAME')[..., 0]

        # Cut tapered edges.
        lut_cen = np.arange(lut_num) + (lut_draw - lut_num) // 2
        lut = tf.gather(lut, indices=lut_cen, axis=1)

        # Normalize and apply.
        lut_norm = KL.Lambda(lambda x: tf.map_fn(ne.utils.minmax_norm, x) * lut_max)
        lut, image = map(lut_norm, (lut, image))
        lut_func = lambda x: tf.gather(x[0], indices=tf.cast(x[1], integer_type), axis=0)
        image = KL.Lambda(
            lambda x: tf.map_fn(lut_func, x, fn_output_signature=compute_type),
        )([lut, image])

    # Bias field.
    if bias_first and bias_max > 0:
        bias_field = ne.layers.PerlinNoise(
            noise_min=bias_min,
            noise_max=bias_max,
            isotropic=False,
            fwhm_min=bias_blur_min / (2 if half_res else 1),
            fwhm_max=bias_blur_max / (2 if half_res else 1),
            reduce=tf.math.reduce_max,
            dtype=compute_type,
            seed=seeds.pop('bias', None),
        )(image)
        bias_field = KL.Lambda(bias_func, name=f'bias_func_{id}')(bias_field)
        image *= bias_field

    # Noise.
    image = ne.layers.GaussianNoise(noise_min, noise_max, seed=seeds.pop('noise', None))(image)

    # Blur.
    image = ne.layers.GaussianBlur(
        sigma=blur_max,
        min_sigma=blur_min,
        random=True,
        seed=seeds.pop('blur', None),
    )(image)

    # Bias field.
    if not bias_first and bias_max > 0:
        bias_field = ne.layers.PerlinNoise(
            noise_min=bias_min,
            noise_max=bias_max,
            isotropic=False,
            fwhm_min=bias_blur_min / (2 if half_res else 1),
            fwhm_max=bias_blur_max / (2 if half_res else 1),
            reduce=tf.math.reduce_max,
            dtype=compute_type,
            seed=seeds.pop('bias', None),
        )(image)
        bias_field = KL.Lambda(bias_func, name=f'bias_func_{id}')(bias_field)
        image *= bias_field

    # Create thick slices.
    image = ne.layers.Subsample(
        prob=slice_prob,
        stride_min=max(1, slice_stride_min / (2 if half_res else 1)),
        stride_max=max(1, slice_stride_max / (2 if half_res else 1)),
        axes=slice_axes,
        seed=seeds.pop('slice', None),
    )(image)

    # Intensity manipulations.
    image = ne.layers.RandomClip(
        clip_min=0,
        clip_max=clip_max,
        prob_min=1 if clip_max else 0,
        prob_max=1 if clip_max else 0,
        axes=(0, -1),
    )(image)
    if normalize:
        image = KL.Lambda(lambda x: tf.map_fn(ne.utils.minmax_norm, x))(image)
    if gamma > 0:
        assert 0 < gamma < 1, f'gamma value {gamma} outside interval [0, 1)'
        gamma = tf.random.uniform(
            shape=(batch_size, *[1] * num_dim, num_chan),
            minval=1 - gamma,
            maxval=1 + gamma,
            dtype=image.dtype,
            seed=seeds.pop('gamma', None),
        )
        image = tf.pow(image, gamma)

    outputs = [image]
    if return_vel:
        outputs.append(vel_field)
    if return_def:
        outputs.append(def_field)
    if return_aff:
        outputs.append(affine)
    if return_composed:
        outputs.append(trans)

    assert not seeds, f'unknown seeds {seeds}'
    return tf.keras.Model(input_model.inputs, outputs)


def labels_to_labels(
    labels_in,
    in_shape=None,
    input_model=None,
    background=0,
    shapes_add=True,
    shapes_num=8,
    shapes_zero_max=0.0,
    shapes_fwhm_min=16,
    shapes_fwhm_max=64,
    shapes_warp_max=0.1,
    shapes_gen_scale=2,
    seeds={},
):
    '''
    Generate N random shapes and replace the background of a label map. If the largest label in
    the input label maps is L, the added labels will be {L+1, L+2, ...L+N}.

    This is work in progress and should not be shared externally. If you would
    like to use the code for an abstract or paper, please contact the author.

    Author:
        mu40

    Arguments:
        labels_in:
            List of all possible labels of the input label maps.
        in_shape:
            Spatial dimensions of the input label map.
        input_model:
            Model whose outputs will be used as data inputs, and whose
            inputs will be used as inputs to the returned model.
        background:
            Background label to replace.
        shapes_add:
            Replace the background with random shapes.
        shapes_num:
            Number of random shapes N to generate.
        shapes_zero_max:
            Maximum proportion of shapes randomly set to zero.
        shapes_fwhm_min:
            Minimum blurring FWHM for shapes creation.
        shapes_fwhm_max:
            Maximum blurring FWHM for shapes creation.
        shapes_warp_max:
            Maximum warp SD for shapes creation as a proportion of the tensor size.
        shapes_gen_scale:
            Scale at which shapes are generated. A value of 2 means half resolution.
        seeds:
            Dictionary of seeds for reproducible randomization.

    Returns:
        Model for modifying input label maps.
    '''
    # Compute type.
    compute_type = tf.keras.mixed_precision.global_policy().compute_dtype
    compute_type = tf.dtypes.as_dtype(compute_type)
    integer_type = tf.int32

    # Input model.
    if input_model is None:
        labels = KL.Input(shape=(*in_shape, 1), name=f'input_{id}', dtype=compute_type)
        input_model = tf.keras.Model(*[labels] * 2)
    labels = input_model.output
    if not labels.dtype.is_floating:
        labels = tf.cast(labels, compute_type)

    # Dimensions.
    in_shape = np.asarray(labels.shape[1:-1])
    num_dim = len(in_shape)
    batch_size = tf.shape(labels)[0]

    shapes_draw = lambda x: utils.generative.draw_shapes(
        shape=in_shape,
        num_label=shapes_num,
        zero_max=shapes_zero_max,
        fwhm_min=shapes_fwhm_min,
        fwhm_max=shapes_fwhm_max,
        warp_max=shapes_warp_max,
        gen_scale=shapes_gen_scale,
        dtype=x.dtype,
        seed=seeds.get('shapes'),
    )
    shapes = KL.Lambda(lambda x: tf.map_fn(shapes_draw, x))(labels)

    # Offset and merge.
    offset = tf.reduce_max(labels_in) + 1
    if offset.dtype != labels.dtype:
        offset = tf.cast(offset, labels.dtype)
    shapes += offset
    background = tf.equal(labels, background)
    labels += tf.cast(background, labels.dtype) * shapes

    return tf.keras.Model(input_model.inputs, outputs=labels)


class RigidMotionSynth(ne.tf.modelio.LoadableModel):
    """
    [Experimental] rigid motion simulation.

    Author: adalca
    """

    @ne.tf.modelio.store_config_args
    def __init__(self,
                 inshape,
                 **kwargs):
        """
        Parameters:
            inshape: Input shape, e.g. (160, 160, 192).
            **kwargs: arguments passed to layer RigidMotionSynth

        Authors: adalca
        """

        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        inshape = tuple(inshape)
        inp = KL.Input(inshape + (1,))
        out = vxms.layers.RigidMotionSynth(**kwargs)(inp)

        super().__init__(inputs=inp, outputs=out)

        # cache pointers to important layers and tensors for future reference
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        # TODO: add references to motion produced from layer


def _remove_skull(mri, skull_labels=None):
    '''helper function to remove non-brain labels from a label volume
    '''
    if skull_labels is None:
        skull_labels = [259, 258, 165]

    dtype = mri.dtype
    inshape = tf.shape(mri)
    if mri.get_shape().as_list()[-1] > 1:
        mri = tf.cast(tf.math.argmax(mri, axis=-1), dtype=dtype)[..., tf.newaxis]
        inshape = tf.shape(mri)

    skull_mask = tf.ones(inshape, tf.bool)
    for label in skull_labels:
        lmask = tf.where(
            tf.equal(mri, label), tf.zeros(inshape, tf.bool), tf.ones(inshape, tf.bool))
        skull_mask = tf.logical_and(skull_mask, lmask)

    mri *= tf.cast(skull_mask, dtype)

    return mri


def SynthAtrophyPair(inshape, labels_in, labels_out, 
                     name='synth_atropy_pair', 
                     gen_args={}, 
                     structure_list={},  
                     same_contrast=False,
                     max_trans=5,
                     warp_std=1,
                     warp_res=16,
                     max_rot=10,
                     max_scale=None,
                     max_shear=None,
                     debug_outputs=False,
                     fill_value=0,
                     max_lesions_to_insert=0,
                     lesion_label=77,        # only used if max_lesions_to_insert > 0
                     unique_label=100,       # only used if max_lesions_to_insert > 0
                     insert_labels=[2, 41],  # only used if max_lesions_to_insert > 0
                     interior_dist=3,        # only used if max_lesions_to_insert > 0
                     max_label_vol=20,       # only used if max_lesions_to_insert > 0
                     subsample_atrophy=.5,
                     remove_skull=True,
                     one_hot=False,          # passed to labels_to_image
                     mapping=None,
                     return_warp=False):
    '''Build model that augments label maps and synthesizes images from them.
    Author:
       brf2

    Arguments:
        inshape        - shape of input without channels
        labels_in      - see nes.models.labels_to_image
        labels_out     - see nes.models.labels_to_image
        name           - name of the model
        gen_args       - see nes.models.labels_to_image
        structure_list - see vxms.layers.resize_labels
        same_contrast  - boolean. If True paired images will have the same contrast (but still
                         different across the batch or in different batches)
        max_trans      - max translation for augmentation
        warp_std       - warp std dev for warping of image to simulate e.g. grad nonlinearities (1)
        warp_res       - warp resolution for image warp to simulate e.g. grad nonlinearities (16)
        max_rot        - maximum rotation for rigid augmentation (10)
        max_scale      - maximum scale for affine augmentation (10)
        max_shear      - maximum shear for affine augmentation (10)
        debug_outputs  - boolean (False) to return a bunch of debugging outputs
        fill_value     - fill value for augmentation transforms (0)
        max_lesions_to_insert - # of lesion labels to synthesize
        subsample_atrophy - see vxms.layers.resize_labels
        return_warp    - return warp that maps t1 images to t2
        one_hot=False  - passed to labels_to_image

    Returns:
      [0,1] the intensity inputs (t1, t2 with atrophy, warped)
      [2,3] the masked intensity images (t1, t2 no atrophy)
      [4] the masked label map, no atrophy, no warping
      [5] the t2 label map with atrophy (unwarped)
      [6] the warped, masked t2 label map, no atrophy
      [7] the t2 label map with atrophy, warped (so this one minus the previous is the change map)
      [8] the t2 image without atrophy warped into time 2 coords
      [9] the t2 label map without atrophy and unwarped (suitable for rigid reg loss function)
    '''

    seeds_t1 = dict(mean=1, std=2, warp=3, blur=4, bias=5, gamma=6)  
    seeds_t2 = dict(mean=1, std=7, warp=3, blur=5, bias=6, gamma=5)   # same mean and warp as t1
    seeds_t2_wa = seeds_t2.copy()
    seeds_warp = dict(mean=2, std=3, warp=4, blur=5, bias=6, gamma=7)

    # these two models create images with the same warp/contrast
    # the second one will have atrophy added
    num_chan = 1 if same_contrast else 2
    if debug_outputs:
        num_chan = 2
    gen_model_t1 = labels_to_image(
        labels_in=labels_in,
        in_shape=inshape,
        labels_out=labels_out,
        id=0,
        num_chan=num_chan,
        return_def=False,
        one_hot=one_hot,
        seeds=seeds_t1,
        **gen_args
    )
    gen_model_t2 = labels_to_image(
        labels_in=labels_in,
        in_shape=inshape,
        labels_out=labels_out,
        id=1,
        num_chan=num_chan,
        return_def=False,
        one_hot=one_hot,
        seeds=seeds_t2,         # same seeds as gen_model so intensity distributions are same
        **gen_args
    )
    gen_model_t2_wa = labels_to_image(
        labels_in=labels_in,
        in_shape=inshape,
        labels_out=labels_out,
        id=2,
        num_chan=num_chan,
        return_def=False,
        one_hot=one_hot,
        seeds=seeds_t2_wa,         # same seeds as gen_model so intensity distributions are same
        **gen_args
    )

    # this model will take the label with atrophy added and include a bit of geometric
    # distortion to it
    if warp_std > 0:
        warp_model = image_to_image(
            in_shape=inshape,
            warp_res=warp_res,
            warp_max=warp_std,
            warp_zero_mean=False,   # allow some mistakes in rigid reg
            id=3,
            seeds=seeds_warp,         
            bias_max=0,
            blur_max=0,
            lut_blur_max=0,
            noise_max=0,
            gamma=0,
            normalize=True,
            return_def=True,
        )
    else:
        warp_model = None

    label_input = KL.Input(shape=tuple(inshape) + (1,), name='label_in')
    # print(f'eager is {tf.executing_eagerly()}')
    if max_lesions_to_insert > 0:
        # print(f'adding max {max_lesions_to_insert} in atrophy synth')
        labels_with_lesions = vxms.layers.AddLesions(
            lesion_label=lesion_label, 
            unique_label=unique_label, 
            insert_labels=insert_labels, 
            interior_dist=interior_dist, 
            max_labels=max_lesions_to_insert, 
            max_label_vol=max_label_vol,
            name='AddLesions')(label_input)
    else:
        labels_with_lesions = label_input   # lesions are disabled

    label_input_wa = vxms.layers.ResizeLabels(
        structure_list, name='resize_labels', subsample_atrophy=subsample_atrophy)(
            labels_with_lesions)

    # synthesize two images with the same basic contrast and identical geometry as well as
    # a label map 
    synth_image_t1, synth_labels = gen_model_t1(labels_with_lesions)
    synth_image_t2, synth_labels2 = gen_model_t2(labels_with_lesions)

    # synthesize an image and associated label map that adds some atrophy, 
    # otherwised matched to prev t2
    synth_image_t2_wa, synth_labels_wa = gen_model_t2_wa(label_input_wa)  # with lesions and atrophy

    # extract the separate image contrasts 
    synth_image_t1 = KL.Lambda(lambda x: x[..., 0:1], name='synth_image_t1')(synth_image_t1)

    i0 = 0 if same_contrast else 1
    synth_image_t2 = KL.Lambda(lambda x: x[..., i0:i0 + 1], name='synth_image_t2')(synth_image_t2) 
    synth_image_t2_wa = KL.Lambda(lambda x: x[..., i0:i0 + 1], 
                                  name='synth_image2_wa')(synth_image_t2_wa)
    synth_labels = tf.cast(synth_labels, tf.float32)
    synth_labels_wa = tf.cast(synth_labels_wa, tf.float32)

    # a note on naming:
    # si_ tensors are images
    # sl_ tensors are label maps
    # _wa indicates with atrophy
    # map the image and labels to two new positions in space (rigid)
    si_t1, affine_t1 = vxms.layers.AffineAugment(
        name='si_t1', max_trans=max_trans, max_rot=max_rot, 
        max_scale=max_scale, max_shear=max_shear,
        return_mats=True, limit_to_fov=True)([synth_image_t1])
    si_t2, affine_t2 = vxms.layers.AffineAugment(
        name='si_t2', max_trans=max_trans, max_rot=max_rot, 
        max_scale=max_scale, max_shear=max_shear,
        return_mats=True, limit_to_fov=True)([synth_image_t2])

    if mapping is not None:
        synth_labels = KL.Lambda(lambda x: ne.utils.seg.recode(tf.cast(x, tf.int32), mapping))(
            synth_labels)
        synth_labels_wa = KL.Lambda(lambda x: ne.utils.seg.recode(tf.cast(x, tf.int32), mapping))(
            synth_labels_wa)

    # apply the affine transforms to the label maps
    linterp = 'linear' if one_hot else 'nearest'
    sl_t1 = vxm.layers.SpatialTransformer(
        name='sl_t1', interp_method=linterp, 
        fill_value=fill_value)([synth_labels, affine_t1])
    sl_t2 = vxm.layers.SpatialTransformer(
        name='sl_t2', interp_method=linterp, 
        fill_value=fill_value)([synth_labels, affine_t2])
    si_t2_wa = vxm.layers.SpatialTransformer(
        name='si_t2_wa', interp_method='linear', 
        fill_value=fill_value)([synth_image_t2_wa, affine_t2])
    sl_t2_wa = vxm.layers.SpatialTransformer(
        name='sl_t2_wa', interp_method=linterp, 
        fill_value=fill_value)([synth_labels_wa, affine_t2])

    # now add some "MR" distortions 
    if warp_model is not None:
        si_t2_warped, warp = warp_model(si_t2) 
        si_t2_warped = vxm.layers.SpatialTransformer(
            name='si_t2_warped', interp_method='linear', 
            fill_value=fill_value)([si_t2, warp])
        sl_t2_warped = vxm.layers.SpatialTransformer(
            name='sl_t2_warped', interp_method=linterp, 
            fill_value=fill_value)([sl_t2, warp])
        si_t2_wa_warped = vxm.layers.SpatialTransformer(
            name='si_t2_wa_warped', interp_method='linear', 
            fill_value=fill_value)([si_t2_wa, warp])
        sl_t2_wa_warped = vxm.layers.SpatialTransformer(
            name='sl_t2_wa_warped', interp_method=linterp, 
            fill_value=fill_value)([sl_t2_wa, warp])
    else:
        si_t2_warped = si_t2
        sl_t2_warped = sl_t2
        si_t2_wa_warped = si_t2_wa
        sl_t2_wa_warped = sl_t2_wa

    # change things to float32 and create brain mask
    if remove_skull:
        if one_hot:
            warnings.warn('are you sure you want to remove the skull"\
            " with one_hot=True in SynthAtrophyPair???')

        brain_mask_t1 = tf.cast(_remove_skull(sl_t1) > 0, tf.float32)
        brain_mask_t2 = tf.cast(_remove_skull(sl_t2) > 0, tf.float32)
        brain_mask_t2_wa = tf.cast(_remove_skull(sl_t2_wa) > 0, tf.float32)
        brain_mask_t2_warped = tf.cast(_remove_skull(sl_t2_warped) > 0, tf.float32)
        brain_mask_t2_wa_warped = tf.cast(_remove_skull(sl_t2_wa_warped) > 0, tf.float32)

        # skull removed vols without the warping
        si_t1_m = KL.Multiply(name='si_t1_m')([si_t1, brain_mask_t1])
        sl_t1_m = KL.Multiply(name='sl_t1_m')([sl_t1, brain_mask_t1])
        si_t2_m = KL.Multiply(name='si_t2_m')([si_t2, brain_mask_t2])
        sl_t2_m = KL.Multiply(name='sl_t2_m')([sl_t2, brain_mask_t2])
        sl_t2_wa_m = KL.Multiply(name='sl_t2_wa_m')([sl_t2_wa, brain_mask_t2_wa])

        # skull removed with warping
        sl_t2_warped_m = KL.Multiply(name='sl_t2_warped_m')([sl_t2_warped, brain_mask_t2_warped])
        sl_t2_wa_warped_m = KL.Multiply(name='sl_t2_wa_warped_m')(
            [sl_t2_wa_warped, brain_mask_t2_wa_warped])
    else:
        si_t1_m = si_t1
        sl_t1_m = sl_t1
        si_t2_m = si_t2
        sl_t2_m = sl_t2
        sl_t2_wa_m = sl_t2_wa

        # skull removed with warping
        sl_t2_warped_m = sl_t2_warped
        sl_t2_wa_warped_m = sl_t2_wa_warped

    inputs = label_input
    # output the two synthetic images that would be the observed inputs to a regnet and the two
    # images that would be in the loss (and their accompanying label maps)
    # for vxm_dense training inputs will be outputs 0 and 1
    # and loss will take out[4] (masked t1 label map) and out[6] 
    # (warped masked t2 label map with no atrophy added)
    # outputs are:
    # [0,1] the intensity inputs (t1, t2 with atrophy, warped)
    # [2,3] the masked intensity images (t1, t2 no atrophy)
    # [4] the masked label map, no atrophy, no warping
    # [5] the t2 label map with atrophy (unwarped)
    # [6] the warped, masked t2 label map, no atrophy
    # [7] the t2 label map with atrophy, warped (so this one minus the previous is the change map)
    # [8] the t2 image without atrophy warped into time 2 coords
    # [9] the t2 label map without atrophy and unwarped (suitable for rigid reg loss function)
    outputs = [
        si_t1,              # 0 
        si_t2_wa_warped,    # 1
        si_t1_m,            # 2
        si_t2_m,            # 3
        sl_t1_m,            # 4
        sl_t2_wa_m,         # 5
        sl_t2_warped_m,     # 6
        sl_t2_wa_warped_m,  # 7
        #        si_t2_warped,       # 8
        #        sl_t2_m]            # 9
    ]

    if return_warp:
        inv_aff1 = vxm.layers.InvertAffine(name='inv_aff1')(affine_t1)
        warp_t1_to_t2 = vxm.layers.ComposeTransform(name='warp_t1_to_t2')(
            [inv_aff1, affine_t2, warp])
        outputs += [warp_t1_to_t2]   # 10 - warp that takes t1->t2

    if debug_outputs:
        outputs += [
            synth_image_t1,      # 10, 8
            synth_labels,        # 11, 9
            synth_image_t2,      # 12, 10
            synth_image_t2_wa,   # 13, 11
            synth_labels_wa,     # 14, 12
            sl_t1,               # 15, 13
            si_t2,               # 16, 14
            sl_t2,               # 17, 15
            si_t2_wa,            # 18, 16
            synth_labels2        # 19, 17
        ]
        #  si_t2_wa, sl_t2_wa, si_t2_warped, sl_t2_warped, si_t1_m, sl_t1_m, 
        #  si_t2_m, sl_t2_m]

    model = tf.keras.Model(inputs, outputs)
    return model


###############################################################################
# Helper function
###############################################################################
