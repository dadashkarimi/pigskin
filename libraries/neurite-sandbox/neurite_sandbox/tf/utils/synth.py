import numpy as np
import tensorflow as tf

import neurite as ne
import voxelmorph as vxm


def perlin_nshot_task(in_shape,
                      num_gen,
                      num_label=16,
                      shapes_im_scales=(32, 64),
                      shapes_warp_scales=(16, 32, 64),
                      shapes_im_max_std=1,
                      shapes_warp_max_std=8,
                      visualize=False,
                      min_int=0,
                      max_int=1,
                      std_min=0.01,
                      std_max=0.2,
                      lab_int_interimage_std=0.01,
                      warp_std=2,
                      warp_res=(8, 16, 32),
                      bias_res=40,
                      bias_std=0.3,
                      blur_std=1.5):
    """ 
    generate nshot task from perlin-noise-based shapes

    Universeg TODO notes:
    # - randomly sample parameters to this method to see what tasks look like. (within reason)
    # - there are other parameters in vxm.models.labels_to_image that we should explore

    to visualize, use 
    ne.plot.slices(images);
    ne.plot.slices(slices, cmaps=['tab20c']);

    Args:
        in_shape (tuple): e.g. (128,) * 2,  # Input shapes
        num_gen (int): number of images to generate
        num_label (int): number of labels. e.g. 16.
        shapes_im_scales (tuple, optional): _description_. Defaults to (32, 64).
        shapes_warp_scales (tuple, optional): _description_. Defaults to (16, 32, 64).
        shapes_im_max_std (int, optional): _description_. Defaults to 1.
        shapes_warp_max_std (int, optional): _description_. Defaults to 8.
        visualize (bool, optional): _description_. Defaults to False.
        min_int (int, optional): minimum image intensity. Defaults to 0.
        max_int (int, optional): maximum image intensity. Defaults to 1.
        std_min (float, optional): minimum image noise. Defaults to 0.01.
        std_max (float, optional): maximum image noise. Defaults to 0.2.
        lab_int_interimage_std (float, optional): images get slightly different shape mean. 
           This is the maximum standard deviation of a noise between these intensities 
           (same shape, diff images). Keep this small to keep contrast among nshot images.
           Defaults to 0.01.
        warp_res (tuple, optional): resolutions to generate random warp. Defaults to (8, 16, 32).
        bias_res (int, optional): resolutions to generate random bias. Defaults to 40.
        bias_std (float, optional): bias noise. Defaults to 0.3.
        blur_std (float, optional): blur smoothness. Defaults to 1.5.

    Returns:
        _type_: _description_
    """

    im, lab = perlin_shapes_image(in_shape=in_shape,
                                  num_label=num_label,
                                  im_scales=shapes_im_scales,
                                  warp_scales=shapes_warp_scales,
                                  im_max_std=shapes_im_max_std,
                                  warp_max_std=shapes_warp_max_std,
                                  visualize=visualize)

    images, label_maps = labels_to_images(lab,
                                          num_gen,
                                          min_int=min_int,
                                          max_int=max_int,
                                          std_min=std_min,
                                          std_max=std_max,
                                          lab_int_interimage_std=lab_int_interimage_std,
                                          warp_std=warp_std,
                                          warp_res=warp_res,
                                          bias_res=bias_res,
                                          bias_std=bias_std,
                                          blur_std=blur_std)

    # select random
    return images, label_maps


def perlin_shapes_image(in_shape=(128,) * 2,  # Input shapes
                        num_label=16,
                        im_scales=(32, 64),
                        warp_scales=(16, 32, 64),
                        im_max_std=1,
                        warp_max_std=8,
                        visualize=False):
    """ 
    perlin shape image

    Author:
        avd12

    Args:
        in_shape (_type_, optional): _description_. Defaults to (128,)*2.
        im_scales (tuple, optional): _description_. Defaults to (32, 64).
        warp_scales (tuple, optional): _description_. Defaults to (16, 32, 64).
        im_max_std (int, optional): _description_. Defaults to 1.
        warp_max_std (int, optional): _description_. Defaults to 8.
        visualize (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    num_dim = len(in_shape)

    # Draw image and warp.
    im = ne.utils.augment.draw_perlin(
        out_shape=(*in_shape, num_label),
        scales=im_scales, max_std=im_max_std,
    )
    warp = ne.utils.augment.draw_perlin(
        out_shape=(*in_shape, num_label, num_dim),
        scales=warp_scales, max_std=warp_max_std,
    )

    # Transform and create label map.
    im = vxm.utils.transform(im, warp)
    lab = np.uint8(tf.argmax(im, axis=-1))

    if visualize:
        ne.plot.slices(lab, cmaps=['tab20c'], width=2.5)

    return im, lab


def labels_to_images(lab,
                     num_gen,
                     min_int=0,
                     max_int=1,
                     std_min=0.01,
                     std_max=0.2,
                     lab_int_interimage_std=0.01,
                     warp_std=2,
                     warp_res=(8, 16, 32),
                     bias_res=40,
                     bias_std=0.3,
                     blur_std=1.5):
    """_summary_

    Author:
        avd12

    Args:
        lab (np.array): label image
        num_gen (int): number of images to generate. We'll probably want closer to 100 at least.
        min_int (int, optional): minimum image intensity. Defaults to 0.
        max_int (int, optional): maximum image intensity. Defaults to 1.
        std_min (float, optional): minimum image noise. Defaults to 0.01.
        std_max (float, optional): maximum image noise. Defaults to 0.2.
        lab_int_interimage_std (float, optional): images get slightly different shape mean.
           This is the maximum standard deviation of a noise between these intensities 
           (same shape, diff images). Keep this small to keep contrast among nshot images.
           Defaults to 0.01.
        warp_res (tuple, optional): resolutions to generate random warp. Defaults to (8, 16, 32).
        bias_res (int, optional): resolutions to generate random bias. Defaults to 40.
        bias_std (float, optional): bias noise. Defaults to 0.3.
        blur_std (float, optional): blur smoothness. Defaults to 1.5.

    Returns:
        _type_: _description_
    """
    num_label = int(lab.max() + 1)
    in_shape = lab.shape

    # get random intensities and noise
    mean_min = np.random.uniform(min_int, max_int, size=num_label)
    mean_max = mean_min + np.random.normal(0, lab_int_interimage_std, size=num_label)
    mean_max = mean_max.clip(min_int, max_int)

    # Image generation.
    gen_arg = dict(
        in_shape=in_shape,
        in_label_list=np.arange(num_label),
        warp_std=warp_std,
        warp_res=warp_res,
        mean_min=mean_min,
        mean_max=mean_max,
        std_min=std_min,
        std_max=std_max,
        bias_res=bias_res,
        bias_std=bias_std,
        blur_std=blur_std
    )
    gen_model = ne.models.labels_to_image(**gen_arg, id=1)

    input = np.expand_dims(lab, axis=(0, -1))
    data = [gen_model.predict(input, verbose=0) for _ in range(num_gen)]
    # each data [img, seg]

    images = [f[0].squeeze() for f in data]
    label_maps = [f[1].squeeze() for f in data]

    return images, label_maps


def link_neuro_labels(labels_in, lut=None):
    ''' 
    build a dictionary of linkages to pass to labels_to_image that constrains e.g. left
    labels to have the same intensity as right and all vents to look the same
    assumes FS aseg labels unless a lut is specified
    '''
    labels_in = {int(lab): int(lab) for lab in labels_in}

    if lut is None:   # assume aseg labels
        labels_in[2] = 41
        labels_in[3] = 42
        labels_in[4] = 43
        labels_in[5] = 43
        labels_in[44] = 43
        labels_in[7] = 46
        labels_in[8] = 47
        labels_in[10] = 49
        labels_in[11] = 50
        labels_in[12] = 51
        labels_in[13] = 52
        labels_in[14] = 43
        labels_in[15] = 43
        labels_in[17] = 53
        labels_in[18] = 53
        labels_in[54] = 53
    else:
        left_wm = lut.search('Left-Cerebral-White-Matter')[0]
        right_wm = lut.search('Right-Cerebral-White-Matter')[0]
        left_gm = lut.search('Left-Cerebral-Cortex')[0]
        right_gm = lut.search('Right-Cerebral-Cortex')[0]
        left_hippo = lut.search('Left-Hippocampus')[0]
        right_hippo = lut.search('Right-Hippocampus')[0]
        left_amy = lut.search('Left-Amygdala')[0]
        right_amy = lut.search('Right-Amygdala')[0]
        left_caudate = lut.search('Left-Caudate')[0]
        right_caudate = lut.search('Right-Caudate')[0]
        left_putamen = lut.search('Left-Putamen')[0]
        right_putamen = lut.search('Right-Putamen')[0]
        left_pallidum = lut.search('Left-Pallidum')[0]
        right_pallidum = lut.search('Right-Pallidum')[0]
        left_vent = lut.search('Left-Lateral-Ventricle')[0]
        right_vent = lut.search('Right-Lateral-Ventricle')[0]
        left_inf_lat_vent = lut.search('Left-Inf-Lat-Ven')[0]
        right_inf_lat_vent = lut.search('Right-Inf-Lat-Ven')[0]
        third_vent = lut.search('3rd-Ventricle')[0]
        fourth_vent = lut.search('4th-Ventricle')[0]
        left_thalamus = lut.search('Left-Thalamus')[0]
        right_thalamus = lut.search('Right-Thalamus')[0]
        left_accumbens = lut.search('Left-Accumbens')[0]
        right_accumbens = lut.search('Right-Accumbens')[0]
        left_cbm_wm = lut.search('Left-Cerebellum-White-Matter')[0]
        right_cbm_wm = lut.search('Right-Cerebellum-White-Matter')[0]
        left_cbm_gm = lut.search('Left-Cerebellum-Cortex')[0]
        right_cbm_gm = lut.search('Right-Cerebellum-Cortex')[0]
        labels_in[left_wm] = right_wm
        labels_in[left_gm] = right_gm
        labels_in[left_vent] = right_vent
        labels_in[left_inf_lat_vent] = right_vent
        # labels_in[44] = 4
        labels_in[left_cbm_wm] = right_cbm_wm
        labels_in[left_cbm_gm] = right_cbm_gm
        labels_in[left_thalamus] = right_thalamus
        labels_in[left_caudate] = right_caudate
        labels_in[left_putamen] = right_putamen
        labels_in[left_pallidum] = right_pallidum
        labels_in[third_vent] = right_vent
        labels_in[fourth_vent] = right_vent
        labels_in[left_hippo] = right_hippo
        labels_in[left_amy] = right_hippo
        labels_in[right_amy] = right_hippo

    return labels_in


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
