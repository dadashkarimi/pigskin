import numpy as np
import sys
import warnings
import tensorflow as tf
import neurite as ne
import neurite_sandbox as nes
import voxelmorph as vxm
import voxelmorph_sandbox as vxms


def draw_flip_matrix(grid_shape, shift_center=True, last_row=True, dtype=tf.float32, seed=None):
    warnings.warn('vxms.utils.augment.draw_flip_matrix is deprecated and will be removed in the '
                  'near future. Please use vxm.utils.augment.draw_flip_matrix instead.')
    return vxm.utils.draw_flip_matrix(grid_shape, shift_center, last_row, dtype, seed)


def draw_swap_matrix(num_dim, last_row=True, dtype=tf.float32, seed=None):
    warnings.warn('vxms.utils.augment.draw_swap_matrix is deprecated and will be removed in the '
                  'near future. Please use vxm.utils.augment.draw_swap_matrix instead.')
    return vxm.utils.draw_swap_matrix(num_dim, last_row, dtype, seed)


def draw_affine_matrix(
    shift=None, rotate=None, scale=None, shear=None, deg=True,
    last_row=False, normal=False, is_2d=False, dtype=tf.float32, seed=None,
):
    warnings.warn('vxms.utils.augment.draw_affine_matrix is deprecated and will be removed in the '
                  'near future. Please use vxm.utils.augment.draw_affine_params or '
                  'vxm.layers.DrawAffineParams instead.')
    ndims = 2 if is_2d else 3
    par = vxm.utils.draw_affine_params(shift,
                                       rotate,
                                       scale,
                                       shear,
                                       normal_shift=normal,
                                       normal_rot=normal,
                                       normal_scale=normal,
                                       normal_shear=normal,
                                       shift_scale=False,
                                       ndims=ndims,
                                       concat=True,
                                       dtype=dtype,
                                       seeds={k: seed for k in ('shift', 'rot', 'scale', 'shear')})

    arg = dict(deg=deg, shift_scale=True, last_row=last_row, ndims=ndims)
    return vxm.utils.params_to_affine_matrix(par, **arg)


def draw_b0_distortion(in_shape, read_dir=None, max_std=50, modulate=True, min_blur=8, max_blur=32):
    '''
    Draw a 1D deformation along a direction to simulate MRI-typical B0 distortions in 2D or 3D.

    Parameters:
        in_shape: Image shape without batch or feature dimension as a list, tuple or flat array.
        read_dir: Vector indicating the direction, of the same size as in_shape. Can be a list,
            tuple, array or tensor. None means the direction will be randomized.
        max_std: Maximum standard deviation (SD) of the deformation.
        modulate: Whether to sample the warp SD uniformely from [0, max_std). If False, warps will
            be sampled from a normal distribution of SD max_std.
        min_blur: Minimum SD of the random Gaussian kernel used to blur the deformation.
        max_blur: Maximum SD of the random Gaussian kernel used to blur the deformation.

    Returns:
        Deformation field as a tensor of shape (*in_shape, len(in_shape)).

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    '''
    in_shape = tuple(in_shape)  # change it from list
    ftype = tf.float32
    num_dim = len(in_shape)
    def_shape = (*in_shape, num_dim)
    rand = tf.random.get_global_generator()
    assert num_dim in (2, 3), 'only 2D and 3D supported'

    # Prepare readout direction.
    if read_dir is None:
        read_dir = rand.uniform((num_dim,), minval=1e-6, dtype=ftype)
    if not tf.is_tensor(read_dir) or read_dir.dtype != ftype:
        read_dir = tf.cast(read_dir, dtype=ftype)
    read_dir = tf.reshape(read_dir, shape=(-1,))
    read_dir, _ = tf.linalg.normalize(read_dir)

    # Draw single-channel deformation.
    def_std = rand.uniform((1,), maxval=max_std) if modulate else max_std
    def_field = rand.normal(shape=(*in_shape, 1), stddev=def_std, dtype=ftype)

    # Blur and rescale.
    kernels = ne.utils.gaussian_kernel(max_blur, min_sigma=min_blur, random=True, separate=True)
    before = tf.math.reduce_std(def_field)
    def_field = ne.utils.separable_conv(def_field, [kernels] * num_dim)
    after = tf.math.reduce_std(def_field)
    def_field *= before / after

    # Create rotation matrix.
    if num_dim == 3:
        x = tf.constant((1, 0, 0), ftype)
        z = tf.constant((0, 0, 1), ftype)
        dot = lambda a, b: tf.reduce_sum(a * b)
        proj = read_dir - z * dot(z, read_dir)
        proj, _ = tf.linalg.normalize(proj)
        azim = tf.math.acos(dot(proj, x))
        elev = tf.math.acos(dot(proj, read_dir))
        rot_y = vxm.utils.angles_to_rotation_matrix((0, -elev), deg=False, ndims=num_dim)
        rot_z = vxm.utils.angles_to_rotation_matrix((0, 0, azim), deg=False, ndims=num_dim)
        rot = tf.matmul(rot_z, rot_y)
    else:
        orth = tf.stack((-read_dir[1], read_dir[0]))
        rot = tf.stack((read_dir, orth), axis=-1)

    # Assemble full deformation and rotate.
    if 1:  # brf added (I don't understand the rotation stuff)
        def_vec_field = def_field * tf.ones(in_shape + (3,)) * read_dir
        return def_vec_field
    else:
        def_field = tf.reshape(def_field, (1, -1))
        def_zeros = tf.zeros((num_dim - 1, np.prod(in_shape)), ftype)
        def_field = tf.concat((def_field, def_zeros), axis=0)
        def_field = tf.reshape(def_field, (num_dim, -1))
        def_field = tf.transpose(tf.matmul(rot, def_field))
        return tf.reshape(def_field, (*in_shape, num_dim))


def draw_B0_map(in_shape, max_std=50, modulate=True, 
                min_blur=8, max_blur=32, low_B0_pct=.9, low_B0_std=1):
    '''
    Draw a scalar map to simulate MRI-typical B0 distortions in 2D or 3D.

    Parameters:
        in_shape: Image shape without batch or feature dimension as a list, tuple or flat array.
        max_std: Maximum standard deviation (SD) of the deformation.
        modulate: Whether to sample the warp SD uniformely from [0, max_std). If False, warps will
            be sampled from a normal distribution of SD max_std.
        min_blur: Minimum SD of the random Gaussian kernel used to blur the deformation.
        max_blur: Maximum SD of the random Gaussian kernel used to blur the deformation.

    Returns:
        Deformation field as a tensor of shape (*in_shape, 1).

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    '''
    in_shape = tuple(in_shape)  # change it from list
    ftype = tf.float32
    num_dim = len(in_shape)
    rand = tf.random.get_global_generator()
    assert num_dim in (2, 3), 'only 2D and 3D supported'

    # Prepare readout direction.
    # Draw single-channel deformation.
    low_B0_map = tf.random.uniform((*in_shape, 1))
    high_B0_map = tf.cast(low_B0_map >= low_B0_pct, tf.float32)
    low_B0_map = tf.cast(low_B0_map < low_B0_pct, tf.float32)

    low_B0_std = rand.uniform((1,), maxval=low_B0_std) if modulate else low_B0_std
    B0_field_low = rand.normal(shape=(*in_shape, 1), stddev=low_B0_std, dtype=ftype)
    high_B0_std = rand.uniform((1,), maxval=max_std) if modulate else max_std
    B0_field_high = rand.normal(shape=(*in_shape, 1), stddev=high_B0_std, dtype=ftype)

    B0_field = B0_field_low * low_B0_map + B0_field_high * high_B0_map

    # Blur and rescale.
    kernels = ne.utils.gaussian_kernel(max_blur, min_sigma=min_blur, random=True, separate=True)
    before = tf.math.reduce_std(B0_field)
    B0_field = ne.utils.separable_conv(B0_field, [kernels] * num_dim)
    after = tf.math.reduce_std(B0_field)
    B0_field *= before / after

    return B0_field


def insert_lesions(
        in_label_vol, 
        lesion_label=77, 
        unique_label=100, 
        insert_labels=[2, 41],
        interior_dist=3,
        max_labels=4,
        max_label_vol=20):
    '''
    synthesize lesion labels in the interior of the white matter
    '''
    ftype = tf.float32
    ndim = len(in_label_vol.get_shape().as_list())
    rand = tf.random.get_global_generator()
    prev_label_vol = (in_label_vol)
    nlesions = tf.reshape(rand.uniform((1,), 0, max_labels, dtype=tf.int32), [])

    # print(f'adding {nlesions} lesions (max {max_labels}) into volume with label {lesion_label}')
    # if nlesions == 0:
    #    return in_label_vol
    lesion_vols = rand.uniform((nlesions,), 0, max_label_vol, dtype=tf.int32)
    lesion_hemis = rand.uniform((nlesions,), 0, len(insert_labels), dtype=tf.int32)
    # import pdb as gdb
    # import freesurfer as fs
    # gdb.set_trace()

    interior_label_vols_eroded = []
    for label in insert_labels:
        eroded_vol = nes.utils.utils.morphology_3d(in_label_vol, label, 
                                                   niter=interior_dist, operation='erode',
                                                   eight_connectivity=False, rand_crop=.8)
        interior_label_vols_eroded.append(eroded_vol)

    slist = {unique_label: [0.0, 0.0]}
    prev_label_vol = in_label_vol
    out_label_vol = in_label_vol
    insert_labels = tf.convert_to_tensor(insert_labels, tf.int32)
    interior_label_vols_eroded = tf.convert_to_tensor(interior_label_vols_eroded)
    for lesion_no in range(nlesions):
        # find a single voxel to insert the (unique) lesion label at
        hemi_ind = lesion_hemis[lesion_no]
        hemi_label = insert_labels[hemi_ind]
        tmp_vol = interior_label_vols_eroded[hemi_ind]
        label_inds = tf.random.shuffle(tf.where(tmp_vol))  # tmp_vol already binarized by the erode
        vol_ind = label_inds[0:1]  # randomly pick one voxel
        updates = tf.gather_nd(tf.ones(tmp_vol.shape) * unique_label, vol_ind)
        snd = tf.scatter_nd(vol_ind, updates, tmp_vol.shape)
        mask_vol = tf.cast(tf.equal(snd, 0), ftype)
        label_vol_with_seed = prev_label_vol * mask_vol + snd

        # build a structure list to grow this (unique) label into the hemi label
        slist[unique_label][0] = tf.sqrt(tf.cast(lesion_vols[lesion_no], tf.float32))
        slist[unique_label][1] = tf.cast(hemi_label, tf.float32)
        tmp_vol = resize_labels(label_vol_with_seed, slist, modulate=False)
        new_label_inds = tf.where(tf.equal(tmp_vol, unique_label))
        updates = tf.gather_nd(tf.ones(tmp_vol.shape) * lesion_label, new_label_inds)
        snd = tf.scatter_nd(new_label_inds, updates, tmp_vol.shape)
        mask_vol = tf.cast(tf.equal(snd, 0), prev_label_vol.dtype)
        out_label_vol = prev_label_vol * mask_vol + tf.cast(snd, mask_vol.dtype)
        prev_label_vol = out_label_vol

    # import pdb as gdb
    # gdb.set_trace()
    return tf.cast(out_label_vol, in_label_vol.dtype)


def resize_label_down(in_label_vol, prev_label_vol, label, other_labels, 
                      max_target_vol_change, modulate):
    ftype = prev_label_vol.dtype
    ftype = tf.float32
    tf_false = False
    rand = tf.random.get_global_generator()
    out_label_vol = in_label_vol
    label_inds = tf.where(tf.equal(prev_label_vol, label), name='down_label_inds')
    num_in_struct = tf.cast(tf.shape(label_inds)[0], ftype)

    # build a volume of indices of the labels that are adjacent
    if modulate:
        target_vol_change = rand.uniform((1,), maxval=max_target_vol_change, dtype=ftype)[0]
    else:
        target_vol_change = max_target_vol_change

    if 0:
        num_requested = tf.cast(tf.floor(num_in_struct * target_vol_change), tf.int32)
        num_found = tf.zeros((1,), dtype=tf.int32)
        num = tf.ones((1,), dtype=tf.int32)  # not used first time through
    else:
        num_requested = int(num_in_struct * -target_vol_change)
        num_found = 0
    # print(f'scaling label {label} down by {num_requested} voxels')

    # if label >= 16:
        # print(f'resizing label 77, tvol down change {target_vol_change}, nreq {num_requested}')
        # import pdb 
        # pdb.set_trace()

    niters = 5
    # niters = max(int(target_vol_change*2),5)
    # while num_found < num_requested:
    other_labels = tf.cast(other_labels, prev_label_vol.dtype)
    for iters in range(niters):
        label_vol = tf.equal(prev_label_vol, label)

        # build vols that are binary masks and label indices of other labels
        other_label_mask = tf.zeros(prev_label_vol.shape, dtype=bool)
        other_label_index = tf.zeros(prev_label_vol.shape, dtype=ftype)
        all_other_labels_index = tf.zeros(prev_label_vol.shape, dtype=ftype)
        for other_label in other_labels:
            ovol = nes.utils.morphology_3d(out_label_vol, other_label, niter=1, 
                                           operation='dilate', rand_crop=0.5) 
            # zero other_label_index where the new indices will go
            label_mask = tf.not_equal(ovol, tf.cast(other_label, ovol.dtype))  # only dilated voxels
            other_label_index *= tf.cast(label_mask, other_label_index.dtype)
            # turn binary vol into indices
            other_label_index += tf.cast(ovol, ftype) * tf.cast(other_label, ftype)  
            other_label_mask = tf.equal(all_other_labels_index, 0)  # locations without a value yet
            # only add to zero locations
            all_other_labels_index += other_label_index * tf.cast(other_label_mask, 
                                                                  dtype=other_label_index.dtype)

        # at this point other_label_index is a vol of indices of neighboring labels that will 
        # grow into this one

        # find voxels that border this label and are now one of other_labels
        label_vol_eroded = nes.utils.utils.morphology_3d(
            prev_label_vol, label, niter=1, operation='erode', rand_crop=0.5)
        boundary = tf.logical_and(label_vol, tf.equal(label_vol_eroded, tf_false))
        growth_vol = tf.logical_and(boundary, all_other_labels_index > 0)
        indices = tf.where(growth_vol)
        num = tf.shape(indices)[0]

        cond = tf.greater((num + num_found), num_requested)
        num_thresh = tf.cond(cond, lambda: num_requested - num_found, lambda: num)

        indices = tf.cond(cond, lambda: tf.random.shuffle(indices)[0:num_thresh], lambda: indices)
        num_found = tf.add(num_found, num_thresh, name='down_num_found') 

        # put the ones we found back into the volume to be returned
        updates = tf.gather_nd(all_other_labels_index, indices, name='down_updates')
        snd = tf.scatter_nd(indices, updates, label_vol.shape)
        mask_vol = tf.cast(tf.equal(snd, 0), prev_label_vol.dtype)

        out_label_vol = prev_label_vol * mask_vol + tf.cast(snd, mask_vol.dtype)
        prev_label_vol = out_label_vol  # in case we are not done

    return tf.cast(out_label_vol, in_label_vol.dtype)


# these are not used yet
def up_while_body(x):

    tf_false = False
    prev_label_vol = x[3]
    label = x[4]
    other_labels = x[5]

    label_vol = tf.equal(prev_label_vol, label)

    # build vols that are binary masks and label indices of other labels
    other_label_mask = tf.zeros(prev_label_vol.shape, dtype=bool)
    for other_label in other_labels:
        ovol = tf.equal(prev_label_vol, other_label)
        other_label_mask = tf.math.logical_or(other_label_mask, ovol) 

    # find voxels that border this label and are now one of other_labels
    label_vol_dilated = nes.utils.utils.morphology_3d(
        prev_label_vol, label, niter=1, operation='dilate', rand_crop=0.5)
    tf_false = False
    boundary = tf.math.logical_and(tf.equal(label_vol, tf_false), label_vol_dilated)
    growth_vol = tf.math.logical_and(boundary, other_label_mask)
    indices = tf.where(growth_vol)
    num = tf.shape(indices)[0]

    too_many_indices = (num + num_found) > num_requested
    num = tf.cond(too_many_indices, lambda: num_requested - num_found, lambda: num)
    indices = tf.cond(too_many_indices, lambda: tf.random.shuffle(indices)[0:num], lambda: indices)

    # put the ones we found back into the volume to be returned
    num_found += num
    updates = tf.gather_nd(tf.ones(growth_vol.shape) * label, indices)
    snd = tf.scatter_nd(indices, updates, label_vol.shape)
    mask_vol = tf.cast(tf.equal(snd, 0), ftype)
    out_label_vol = prev_label_vol * mask_vol + snd
    prev_label_vol = out_label_vol  # in case we are not done

    return out_label_vol


def up_while_condition(num, num_found, num_requested, prev_label_vol, label, other_labels):
    return num > 0 and num_found < num_requested


# @tf.function
def resize_label_up(in_label_vol, prev_label_vol, label, other_labels, 
                    max_target_vol_change, modulate):
    tf_false = False
    ftype = tf.float32
    rand = tf.random.get_global_generator()
    out_label_vol = in_label_vol
    label_inds = tf.where(tf.equal(prev_label_vol, label))
    num_in_struct = tf.cast(tf.shape(label_inds)[0], ftype, name='up_num_in_struct')

    # build a volume of indices of the labels that are adjacent
    if modulate:
        target_vol_change = rand.uniform((1,), maxval=max_target_vol_change, dtype=ftype)[0]
    else:
        target_vol_change = max_target_vol_change

    if 0:
        num_requested = tf.cast(tf.floor(num_in_struct * target_vol_change), tf.int32)
        num_found = tf.zeros((1,), dtype=tf.int32)
        num = tf.ones((1,), dtype=tf.int32)  # not used first time through
    else:
        num_requested = int(num_in_struct * target_vol_change)
        num_found = 0

    if label == 77:
        # print(f'resizing label 77, tvol up change {target_vol_change}, nreq {num_requested}')
        import pdb 
        # pdb.set_trace()

    niters = 5
    # niters = max(int(target_vol_change*2),5)
    # print(f'iterating label {label} {niters} times')
    # import pdb as gdb
    # gdb.set_trace()
    # tf.autograph.experimental.set_loop_options(shape_invariants=[(num, tf.shape(num))])
    # while tf.logical_and(tf.greater(num, 0), tf.less(num_found, num_requested)):
    # while num > 0 and num_found < num_requested
    other_labels = tf.cast(other_labels, prev_label_vol.dtype)
    for iters in range(niters):
        label_vol = tf.equal(prev_label_vol, label)

        # build vols that are binary masks and label indices of other labels
        other_label_mask = tf.zeros(prev_label_vol.shape, dtype=bool, name='up_other_label_mask')
        for other_label in other_labels:
            ovol = tf.equal(prev_label_vol, other_label)
            other_label_mask = tf.math.logical_or(other_label_mask, ovol) 

        # find voxels that border this label and are now one of other_labels
        label_vol_dilated = nes.utils.utils.morphology_3d(
            prev_label_vol, label, niter=1, operation='dilate', rand_crop=0.5)
        boundary = tf.math.logical_and(tf.equal(label_vol, tf_false), label_vol_dilated)
        growth_vol = tf.math.logical_and(boundary, other_label_mask)
        indices = tf.where(growth_vol)
        if label == 77:
            # print(f'label 77 iter {iters}: {len(indices)} found')
            import pdb 
            # pdb.set_trace()

        # gdb.set_trace()
        num = tf.shape(indices)[0]
        # if num == 0:
        #    break    # couldn't find any more

        too_many_indices = tf.greater((num + num_found), num_requested)
        num_thresh = tf.cond(too_many_indices, lambda: num_requested - num_found, lambda: num)
        if 0:
            print(num)
            print(num_thresh)
            print(num_requested)
            print(num_found)

        indices = tf.cond(too_many_indices, 
                          lambda: tf.random.shuffle(indices)[0:num_thresh], lambda: indices)
        num_found = tf.add(num_found, num_thresh) 

        # put the ones we found back into the volume to be returned
        updates = tf.gather_nd(tf.ones(growth_vol.shape) * label, indices, name='up_updates')
        snd = tf.scatter_nd(indices, updates, label_vol.shape)
        mask_vol = tf.cast(tf.equal(snd, 0), ftype)
        out_label_vol = prev_label_vol * mask_vol + snd
        prev_label_vol = out_label_vol  # in case we are not done

    # var_list = [num, num_found, num_requested, prev_label_vol, label, other_labels]
    # tf.while_loop(up_while_condition, up_while_body, var_list)
    return out_label_vol


def resize_labels(in_label_vol, structure_list, subsample_atrophy=1, modulate=True):
    '''
    grow or shrink the list of structures provided in the structure_list
    dictionary by the amount specified into the adjacent labels specified
    if modulate is False always use the max amount of vol change requested
    otherwise draw a random uniform volume change in [0, max_target_vol_change]

    Arguments:
       structure_list     - dictionary of allowable label size changes (see below)
       subsample_atrophy  - prob that atrophy of any given label will be added (1 means 
                            it always will be)

    some examples of structure lists:

    shrink hippocampus by 10% with labels 4, 5 and 24 growing into it:
    slist = {
        17 : [-.1, 4, 5, 24]
    } 

    grow wm lesions by 40%:
slist = {
    77 : [.4, 2, 41]
    }

    combine growth and atrophy
slist = {
    17 : [-.1, 4, 5, 24, 0],
    53 : [-.1, 43, 44, 24, 0],
    11 : [-.05, 4, 77],
    77 : [.4, 2]
    }

    '''

    # import pdb as gdb
    # gdb.set_trace()

    # replace 0s in the input volume with some other (unused label) as 0s won't work below
    label_zero = tf.reduce_max(in_label_vol) + 1
    mask = tf.equal(in_label_vol, 0)
    not_mask = tf.cast(tf.logical_not(mask), in_label_vol.dtype)
    mask = tf.cast(mask, dtype=in_label_vol.dtype)
    in_label_vol_nozero = in_label_vol * not_mask + label_zero * mask

    rand = tf.random.get_global_generator()
    out_label_vol = in_label_vol_nozero
    struct_pvals = rand.uniform((len(structure_list),), 0, 1)
    rand = tf.random.get_global_generator()
    for sno, label in enumerate(structure_list):

        # print(f'processing label {label} with max vol change {structure_list[label][0]}')
        do_this_label = tf.cast(tf.less(struct_pvals[sno], subsample_atrophy), 
                                dtype=in_label_vol.dtype, name='do_this_label')
        max_target_vol_change = tf.convert_to_tensor(structure_list[label][0], 
                                                     dtype=tf.float32, 
                                                     name='max_target_vol_change')
        other_labels = tf.convert_to_tensor(structure_list[label][1:], dtype=tf.int32)
        mask = tf.equal(other_labels, 0)
        not_mask = tf.cast(tf.logical_not(mask), other_labels.dtype)
        mask = tf.cast(mask, dtype=other_labels.dtype)
        other_labels = other_labels * not_mask + tf.cast(label_zero, dtype=mask.dtype) * mask

        if label == 77:
            # print(f'resizing label {label} by {max_target_vol_change}, other labels:')
            # print(other_labels)
            import pdb as gdb
            # gdb.set_trace()

        up_fn = lambda: resize_label_up(in_label_vol_nozero, out_label_vol, label, other_labels, 
                                        max_target_vol_change, modulate)
        down_fn = lambda: resize_label_down(in_label_vol_nozero, out_label_vol, label, other_labels, 
                                            max_target_vol_change, modulate)
        changed_label_vol = tf.cond(max_target_vol_change > 0, up_fn, down_fn, 
                                    name='changed_label_vol')
        if 0 and label == 77 and do_this_label:
            import freesurfer as fs
            fv = fs.Freeview()
            fv.vol(out_label_vol, name='out')
            fv.vol(changed_label_vol, name='changed')
            fv.show()
            import pdb 
            # pdb.set_trace()

        # if change label is not true just keep the previous volume
        out_label_vol = changed_label_vol * do_this_label + out_label_vol * (1 - do_this_label)

    # restore the 0s in the output volume
    mask = tf.equal(out_label_vol, label_zero)
    not_mask = tf.cast(tf.logical_not(mask), out_label_vol.dtype)
    mask = tf.cast(mask, out_label_vol.dtype)
    out_label_vol = out_label_vol * not_mask + tf.zeros(tf.shape(mask)) * mask

    return out_label_vol


def resize_labels_old(in_label_vol, structure_list, subsample_atrophy=1, modulate=True):
    '''
    grow or shrink the list of structures provided in the structure_list
    dictionary by the amount specified into the adjacent labels specified
    if modulate is False always use the max amount of vol change requested
    otherwise draw a random uniform volume change in [0, max_target_vol_change]

    Arguments:
       structure_list     - dictionary of allowable label size changes (see below)
       subsample_atrophy  - prob that atrophy of any given label will be added (1 means 
                            it always will be)

    some examples of structure lists:

    shrink hippocampus by 10% with labels 4, 5 and 24 growing into it:
slist = {
    17 : [-.1, 4, 5, 24]
    }

    grow wm lesions by 40%:
slist = {
    77 : [.4, 2, 41]
    }

    combine growth and atrophy
slist = {
    17 : [-.1, 4, 5, 24, 0],
    53 : [-.1, 43, 44, 24, 0],
    11 : [-.05, 4, 77],
    77 : [.4, 2]
    }

    '''

    out_label_vol = in_label_vol
    ftype = tf.float32
    ndim = len(in_label_vol.get_shape().as_list())
    rand = tf.random.get_global_generator()
    prev_label_vol = (in_label_vol)
    struct_pvals = rand.uniform((len(structure_list),), 0, 1)
    print(f'{len(structure_list)} structs, eager is {tf.executing_eagerly()}')
    for sno, label in enumerate(structure_list):
        # if struct_pvals[sno] > subsample_atrophy:
        #     print(f'skipping resizing of struct {sno}, spval is {struct_pvals[sno]}')
        #     continue

        max_target_vol_change = structure_list[label][0]
        other_labels = structure_list[label][1:]  
        label_inds = tf.where(tf.equal(prev_label_vol, label))
        num_in_struct = tf.cast(tf.shape(label_inds)[0], ftype)

        if max_target_vol_change > 0:   # growth
            # build a volume of indices of the labels that are adjacent
            if modulate:
                target_vol_change = rand.uniform((1,), maxval=max_target_vol_change, dtype=ftype)[0]
            else:
                target_vol_change = max_target_vol_change
            num_requested = tf.cast(tf.floor(num_in_struct * target_vol_change), tf.int32)
            num_found = 0
            ino = 0
            while num_found < num_requested:
                label_vol = tf.equal(prev_label_vol, label)

                # build vols that are binary masks and label indices of other labels
                other_label_mask = tf.zeros(in_label_vol.shape, dtype=bool)
                for other_label in other_labels:
                    ovol = tf.equal(prev_label_vol, other_label)
                    other_label_mask = tf.math.logical_or(other_label_mask, ovol) 

                # find voxels that border this label and are now one of other_labels
                label_vol_dilated = nes.utils.utils.morphology_3d(
                    prev_label_vol, label, niter=1, operation='dilate', rand_crop=0.5)
                tf_false = False
                boundary = tf.math.logical_and(tf.equal(label_vol, tf_false), label_vol_dilated)
                growth_vol = tf.math.logical_and(boundary, other_label_mask)
                indices = tf.where(growth_vol)
                num = tf.shape(indices)[0]
                if num == 0:
                    break    # couldn't find any more

                if num + num_found > num_requested:
                    num = num_requested - num_found
                    indices = tf.random.shuffle(indices)[0:num]

                # put the ones we found back into the volume to be returned
                num_found += num
                updates = tf.gather_nd(tf.ones(growth_vol.shape) * label, indices)
                snd = tf.scatter_nd(indices, updates, label_vol.shape)
                mask_vol = tf.cast(tf.equal(snd, 0), ftype)
                out_label_vol = prev_label_vol * mask_vol + snd
                prev_label_vol = out_label_vol  # in case we are not done
                ino += 1
                if ino > 10:
                    break
        else:     # atrophy
            # build a volume of indices of the labels that are adjacent
            if modulate:
                target_vol_change = -rand.uniform(
                    (1,), maxval=-max_target_vol_change, dtype=ftype)[0]
            else:
                target_vol_change = max_target_vol_change
            num_requested = tf.cast(tf.floor(num_in_struct * (-target_vol_change)), tf.int32)
            num_found = 0
            ino = 0
            while num_found < num_requested:
                label_vol = tf.equal(prev_label_vol, label)

                # build vols that are binary masks and label indices of other labels
                other_label_mask = tf.zeros(in_label_vol.shape, dtype=bool)
                other_label_index = tf.zeros(in_label_vol.shape, dtype=ftype)
                for other_label in other_labels:
                    ovol = nes.utils.morphology_3d(out_label_vol, other_label, niter=1, 
                                                   operation='dilate', rand_crop=0.5) 
                    other_label_index += tf.cast(ovol, ftype) * other_label
                    other_label_mask = tf.math.logical_or(other_label_mask, ovol) 

                # find voxels that border this label and are now one of other_labels
                label_vol_eroded = nes.utils.utils.morphology_3d(
                    prev_label_vol, label, niter=1, operation='erode', rand_crop=0.5)
                tf_false = False
                boundary = tf.math.logical_and(label_vol, tf.equal(label_vol_eroded, tf_false))
                growth_vol = tf.math.logical_and(boundary, other_label_mask)
                indices = tf.where(growth_vol)
                num = tf.shape(indices)[0]
                if num == 0:
                    break    # couldn't find any more

                if num + num_found > num_requested:
                    num = num_requested - num_found
                    indices = tf.random.shuffle(indices)[0:num]

                # put the ones we found back into the volume to be returned
                num_found += num
                updates = tf.gather_nd(other_label_index, indices)
                snd = tf.scatter_nd(indices, updates, label_vol.shape)
                mask_vol = tf.cast(tf.equal(snd, 0), ftype)
                out_label_vol = prev_label_vol * mask_vol + snd
                prev_label_vol = out_label_vol  # in case we are not done
                ino += 1
                if ino > 10:
                    break

    return out_label_vol


def draw_crop_mask(x, crop_min=0, crop_max=0.5, axis=None, prob=1, seed=None, bilateral=False):
    warnings.warn('vxms.utils.augment.draw_crop_mask is deprecated and will be removed in the '
                  'near future. Please use ne.utils.augment.draw_crop_mask instead.')
    return ne.utils.augment.draw_crop_mask(x, crop_min, crop_max, axis, prob, bilateral, seed)
