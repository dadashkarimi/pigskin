"""
tools specific to atlas creation
"""

# core
import csv
import os
import itertools
import pathlib

# third party
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
import tensorflow.keras.backend as K
import scipy.ndimage.interpolation
from tqdm import tqdm
import scipy.ndimage
import scipy.stats
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py

# local
import neurite_sandbox as nes
import neurite as ne
import pystrum.pynd as pynd
import imageio
try:
    import cv2
except Exception as _:
    import sys
    print("Issue loading cv2", file=sys.stderr)
import pystrum.pynd.imutils


def celebAMask_HQ_prepare_downsampled_dataset(input_core, output_name, dst_shape,
                                              nb_images=None,
                                              force_grayscale=True,
                                              attr_file=None,
                                              tqdm=tqdm):
    """
    prepare image downsampled dataset and save to h5 for celebAMask-HQ dataset

    examples:
    input_core = pathlib.Path('/nfs02/users/gid-dalcaav/projects/neurite/data/CelebAMask-HQ/orig/')
    output_name = '/nfs02/users/gid-dalcaav/projects/neurite/data/CelebAMask-HQ/proc/proc_128.h5'
    attr_file = input_core/'CelebAMask-HQ-attribute-anno.txt'

    """

    def proc_images(filenames, dst_shape,
                    interpolation_order=1, force_grayscale=False, force_float32=True,
                    tqdm_desc=None, dtype='float32', tqdm_leave=False):
        """
        read, reshape and crop
        """
        nb_images = len(filenames)

        if len(dst_shape) == 3:
            assert dst_shape[-1] == 3 or dst_shape[-1] == 1

        ims = np.zeros((nb_images, *dst_shape), dtype=dtype)
        for i in tqdm(range(nb_images), desc=tqdm_desc, leave=tqdm_leave):
            # im = cv2.cvtColor(cv2.imread(str(filenames[i])), cv2.COLOR_BGR2RGB)
            if not os.path.isfile(str(filenames[i])):
                ims[i, ...] = np.zeros(dst_shape)

            else:
                im = imageio.imread(str(filenames[i]))
                # move to float32
                isint = np.issubdtype(im.dtype, np.integer)

                if force_float32:
                    im = im.astype('float32')

                    if isint:  # if it was int (as opposed to e.g. float64)
                        im = im / 255

                if force_grayscale:
                    im = pynd.imutils.rgb2gray(im)

                # reshape
                reshape = [dst_shape[d] / im.shape[d] for d in range(len(dst_shape))]
                ims[i, ...] = scipy.ndimage.interpolation.zoom(
                    im, reshape, order=interpolation_order).astype(dtype)

        return ims

    # params
    if nb_images is None:
        nb_images = 30000
    grp_factor = 15 / 30000
    input_img = input_core / 'CelebA-HQ-img'
    input_seg = input_core / 'CelebAMask-HQ-mask-anno'

    # get images
    filenames = [input_img / ('%d.jpg' % i) for i in range(nb_images)]
    ds = dst_shape
    if force_grayscale:
        ds = dst_shape[:2]
    ims = proc_images(filenames, ds, interpolation_order=1, force_grayscale=force_grayscale)

    # get segmentation types
    seg_attr = set([f.stem[6:] for f in (input_seg / '0').glob('00000_*.png')])

    segmentations = {}
    for seg_type in tqdm(seg_attr, desc='seg_attr'):
        input_grp = [input_seg / str(int(np.floor(i * grp_factor))) for i in range(nb_images)]
        seg_files = [str(input_grp[i] / '{:05}_{}.png'.format(i, seg_type))
                     for i in range(nb_images)]
        proc = proc_images(seg_files, dst_shape,
                           interpolation_order=0, force_float32=False, tqdm_desc=seg_type,
                           dtype='bool', tqdm_leave=False)
        segmentations[seg_type] = proc[..., 0].astype('bool')

    seg_all = np.zeros([nb_images, *dst_shape[:2]])
    for i in tqdm(range(nb_images), desc='all_seg'):
        seg_all[i, ...] = np.sum(np.stack([segmentations[seg][i, ...]
                                           for seg in seg_attr], 0), 0).clip(0, 1).astype('bool')

    if attr_file is not None:
        y, attr = load_CelebAMask_HQ_attr(attr_file, tqdm=tqdm)

    with h5py.File(output_name, 'w') as f:
        det = f.create_dataset("images", ims.shape, data=ims)
        for seg_type in seg_attr:
            det = f.create_dataset(
                "seg_" + seg_type, segmentations[seg_type].shape, data=segmentations[seg_type])
        det = f.create_dataset("seg_all", seg_all.shape, data=seg_all)

        if attr_file is not None:
            for ai, att in tqdm(enumerate(attr), desc='attributes', leave=True):
                f.create_dataset("attr_" + att, y[:, ai].shape, data=y[:, ai])


def load_CelebAMask_HQ_attr(attr_file, tqdm=tqdm):

    with open(attr_file, 'r') as file:
        #  skip first line
        nb_images = int(file.readline())

        attr = file.readline().split()
        y = np.zeros((nb_images, len(attr)))

        for i in tqdm(range(nb_images), desc='prof_attr_file'):
            y[i, :] = np.stack([int(f) for f in file.readline().split()[1:]], 0)

    return y, attr


def load_CelebAMask(h5_filename, seg=['seg_all']):
    elems = {}
    elems['train'] = slice(0, 20000)
    elems['val'] = slice(20000, 25000)
    elems['test'] = slice(25000, 30000)

    if not isinstance(seg, (list, tuple)):
        seg = [seg]

    with h5py.File(h5_filename, 'r') as file:
        attributes = [f for f in file.keys() if f.startswith('attr_')]
        x = {}
        y = {}
        s = {}

        images = np.array(file['images'])

        # get segs
        all_seg = np.array(file[seg[0]]).astype('bool')
        for se in seg[1:]:
            all_seg = np.logical_or(all_seg, np.array(file[se]).astype('bool'))
        attr = np.stack([np.array(file[f]) for f in attributes], 1)

        for ds in ['train', 'val', 'test']:
            x[ds] = images[elems[ds], ...]
            if x[ds].ndim == 3:
                x[ds] = x[ds][..., np.newaxis]
            s[ds] = all_seg[elems[ds], ...][..., np.newaxis]
            y[ds] = attr[elems[ds], ...]

    return x, s, y, attributes


def load_mnist(tv_ratios=(5 / 6, 1 / 6),
               sel_class=None, pad_amt=0, reshape_fct=None, dataset=mnist):
    """
    load and process mnist for the atlas creation experiments.
    """
    assert 0, 'use ne.tf.data.KerasDataset()'

    x = {}
    y = {}
    (x['train'], y['train']), (x['test'], y['test']) = dataset.load_data()

    x['train'], x['val'] = nes.data.split_dataset(x['train'], ratios=tv_ratios, randomize=False)
    y['train'], y['val'] = nes.data.split_dataset(y['train'], ratios=tv_ratios, randomize=False)

    data_types = ['train', 'test', 'val']

    # select out a class (digit)
    if sel_class is not None:
        for dt in data_types:
            sel_map = y[dt] == sel_class
            x[dt] = x[dt][sel_map, :]
            y[dt] = y[dt][sel_map]

    #  normalize
    for dt in data_types:
        x[dt] = x[dt][..., np.newaxis] / 255

    # pad and reshape
    if pad_amt > 0:
        for dt in data_types:
            x[dt] = np.pad(x[dt], ((0, 0), (pad_amt, pad_amt),
                                   (pad_amt, pad_amt), (0, 0)), mode='constant')
    if reshape_fct is not None:
        res = [1, reshape_fct, reshape_fct, 1]
        for dt in data_types:
            x[dt] = scipy.ndimage.interpolation.zoom(x[dt], res, order=1)

    return x, y


def load_fmnist(tv_ratios=(5 / 6, 1 / 6), sel_class=None, pad_amt=0,
                reshape_fct=None, dataset=fashion_mnist):
    return load_mnist(tv_ratios=tv_ratios, sel_class=sel_class, pad_amt=pad_amt,
                      reshape_fct=reshape_fct, dataset=dataset)


def create_average_atlas(gen, nb_atl_creation=100, tqdm=tqdm, verbose=False):
    """
    create average atlas from generator
    """

    if verbose:
        print('creating "atlas" by averaging %d subjects' % nb_atl_creation)

    rng = range(nb_atl_creation)
    if tqdm is not None:
        rng = tqdm(rng)

    x_avg = 0
    for _ in rng:
        x_avg += next(gen)
    x_avg = x_avg.astype('float')
    x_avg /= nb_atl_creation

    return x_avg


def load_pheno_csv(filename, verbose=True, tqdm=tqdm):
    """
    load a phenotype csv formatted as 
    filename, pheno1, pheno2, etc
    where each pheno is a float
    """

    dct = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        # print header
        header = next(csv_reader)
        if verbose:
            print(header)

        for row in tqdm(csv_reader):
            filename = row[0]
            assert filename not in dct.keys()
            dct[filename] = [float(f) for f in row[1:]]

    return dct


def kde_from_ages(ages, show=0):
    # compute kernel density estimation to enable sampling uniformly across age ranges
    ages_kde = scipy.stats.gaussian_kde(ages)
    prob_ages = ages_kde.evaluate(ages)
    inv_prob_ages = 1 / prob_ages
    inv_prob_ages = inv_prob_ages / np.sum(inv_prob_ages)
    plt.plot(ages, inv_prob_ages, '.')
    plt.title('inverse prob vs age')
    plt.xlabel('age')
    plt.ylabel('1-density')
    return inv_prob_ages


#########################################
# Plotting
#########################################

def plot_atl(atl_model, init_atlas_vol_k, tqdm=tqdm, slice_nr=None,
             classtitles=None, show=True, clip=None, do_colorbars=True, **kwargs):
    """
    this function doesn't make too much sense anymore after it's been stripped down

    atl_model should take in [cond, atlas]
    """

    atlas_pred = atl_model.predict(init_atlas_vol_k)

    # visualize
    if show:
        slices = [f[0, ..., 0] for f in [atlas_pred]]
        if slice_nr is not None:
            slices = [f[..., slice_nr] for f in slices]
        if clip is not None:
            slices = [f.clip(*clip) for f in slices]
        ne.plot.slices(slices, cmaps=['gray'], do_colorbars=do_colorbars, **kwargs)

    return atlas_pred


def plot_cond_atl(atl_model, cond_range, init_atlas_vol_k, tqdm=tqdm, slice_nr=None,
                  classtitles=None, show=True, clip=None, do_colorbars=True, rot90=0, mode='single',
                  **kwargs):
    """
    plot conditional atlas under different conditions

    atl_model should take in [cond, atlas]
    """

    atlas_preds = []
    if tqdm is not None:
        cond_range = tqdm(cond_range)

    for i in cond_range:
        cond = np.array(i).reshape((1, -1,)).astype('float')  # make sure it's a float numpy array
        atlas_preds.append(atl_model.predict([cond, init_atlas_vol_k]))

    # visualize
    if show:

        if classtitles:
            titles = ['%.2f' % f for f in cond_range]
        else:
            titles = ['' for f in cond_range]
        slices = [f[0, ..., 0] for f in atlas_preds]
        if slice_nr is not None:
            slices = [f[..., slice_nr] for f in slices]

        if mode == 'single':
            slices = [np.concatenate([f.squeeze() for f in atlas_preds], 1)]
            titles = None

        if clip is not None:
            slices = [f.clip(*clip) for f in slices]

        ne.plot.slices(slices, cmaps=['gray'], do_colorbars=do_colorbars, titles=titles, **kwargs)

    return atlas_preds


def plot_cond_atl_epochs(atl_model, cond_range, epoch_range, init_atlas_vol_k, filename,
                         tqdm=tqdm, slice_nr=None, mode='single', clip=None, rot90=0, show=True,
                         model_load_weights=None, do_colorbars=False):
    """
    plot conditional atlas under different conditions and epochs

    mode: could be 'single' (as in single large image) or 'separate'
    model_load_weights: a different model to load the weights in, if the model doing the predictions
    has different weights. We are trying to avoid loading by_name, due to sometimes names not being 
    consistent.
    """
    print('Note this modified model weights')
    assert mode in ['single', 'separate'], \
        'mode should be single or separate, got: {}'.format(mode)

    nb_samples = len(epoch_range)
    nb_cond = len(cond_range)

    if model_load_weights is None:
        model_load_weights = atl_model

    # prepare data
    atlas_preds = [[None] * nb_samples for _ in range(nb_cond)]
    titles = [[None] * nb_samples for _ in range(nb_cond)]

    for ei, e in tqdm(enumerate(epoch_range)):
        model_load_weights.load_weights(filename.format(epoch=e))
        for ci, c in tqdm(enumerate(cond_range), leave=False):
            cond = np.array(c).astype(float).reshape((1, -1))
            atlas_preds[ci][ei] = atl_model.predict([cond, init_atlas_vol_k]).squeeze()
            if slice_nr is not None:
                atlas_preds[ci][ei] = atlas_preds[ci][ei][..., slice_nr]
            if rot90 != 0:
                atlas_preds[ci][ei] = np.rot90(atlas_preds[ci][ei], rot90)
            titles[ci][ei] = 'epoch {epoch} cond {cond}'.format(epoch=e, cond=c)

    # combine
    all_pred = []
    for i in range(nb_cond):
        all_pred += atlas_preds[i]

    all_titles = []
    for i in range(nb_cond):
        all_titles += titles[i]

    # plot
    if show:
        if mode == 'single':
            data = [np.concatenate([np.concatenate(f, 0) for f in atlas_preds], 1)]
            if clip is not None:
                data[0] = data[0].clip(*clip)
            ne.plot.slices(data, cmaps=['gray'], do_colorbars=do_colorbars)
        else:
            if clip is not None:
                all_pred = [f.clip(*clip) for f in all_pred]
            ne.plot.slices(all_pred, cmaps=['gray'], do_colorbars=do_colorbars,
                           grid=[nb_cond, nb_samples], titles=all_titles)

    return atlas_preds


def plot_atl_epochs(atl_model, epoch_range, init_atlas_vol_k, filename,
                    tqdm=tqdm, slice_nr=None, mode='single', clip=None,
                    model_load_weights=None, do_colorbars=False):
    """
    plot atlas under different  epochs

    mode: could be 'single' (as in single large image) or 'separate'
    model_load_weights: a different model to load the weights in, if the model doing the predictions
    has different weights. We are trying to avoid loading by_name, due to sometimes names not being 
    consistent.
    """
    print('Note this modified model weights')
    assert mode in ['single', 'separate'], \
        'mode should be single or separate, got: {}'.format(mode)

    nb_samples = len(epoch_range)

    if model_load_weights is None:
        model_load_weights = atl_model

    # prepare data
    atlas_preds = [None] * nb_samples
    titles = [None] * nb_samples

    for ei, e in tqdm(enumerate(epoch_range)):
        model_load_weights.load_weights(filename.format(epoch=e))
        atlas_preds[ei] = atl_model.predict(init_atlas_vol_k).squeeze()
        if slice_nr is not None:
            atlas_preds[ei] = atlas_preds[ei][..., slice_nr]
        titles[ei] = 'epoch {epoch}'

    # combine
    all_pred = atlas_preds
    all_titles = titles

    # plot
    if mode == 'single':
        data = np.concatenate(atlas_preds, 1)
        if clip is not None:
            data[0] = data[0].clip(*clip)
        ne.plot.slices(data, cmaps=['gray'], do_colorbars=do_colorbars)
    else:
        if clip is not None:
            all_pred = [f.clip(*clip) for f in all_pred]
        ne.plot.slices(all_pred, cmaps=['gray'], do_colorbars=do_colorbars,
                       grid=[1, nb_samples], titles=all_titles)

    return atlas_preds


def gather_attribute_stats_neuroimaging(model_atlas_out, inv_flow_model, trf_model_nn,
                                        trf_model_prob, atlas_vol_k, volseg_genobj, age_bins,
                                        # whether to replace age in registration by bin center.
                                        replace_age=True,
                                        tqdm=tqdm, conditional=True, return_seg_prob=True,
                                        nb_subjects=100, des_labels=[4, 17, 43, 53], use_bins=True,
                                        decay=1 / 10):
    """
    nb_ages = 5
    nb_subjects = 100
    des_labels = [4, 17, 43, 53]
    model.load_weights(filename.format(epoch=nb_epochs-1))
    """

    unconditional = not conditional

    def _core_warp(seg, seg_prob, neg_flow):
        # should this even be done?
        warped_subj_seg = trf_model_nn.predict([seg, neg_flow])
        seg_counts = [np.sum(warped_subj_seg == f) for f in des_labels]

        if trf_model_prob.output.shape.as_list()[-1] == 1:
            warped_subj_seg_prob = [trf_model_prob.predict(
                [seg_prob[..., f, np.newaxis], neg_flow]) for f in range(seg_prob.shape[-1])]
            warped_subj_seg_prob = np.concatenate(warped_subj_seg_prob, -1)
        else:
            warped_subj_seg_prob = trf_model_prob.predict([seg_prob, neg_flow])

        # add bg to onehot
        # warped_subj_seg_onehot = np.concatenate([(warped_subj_seg==f).astype(float) for
        #   f in des_labels], -1)
        warped_subj_seg_onehot = warped_subj_seg_prob
        bg = 1 - (np.sum(warped_subj_seg_onehot, -1, keepdims=True) > 0).astype(float)
        all_seg_onehot = np.concatenate([bg, warped_subj_seg_onehot], -1)

        return all_seg_onehot, seg_counts

    # age bins
    # age_bins = np.linspace(np.array(ages).min(), np.array(ages).max(), nb_ages)
    # age_bins = np.linspace(*age_range, nb_ages)
    if conditional:
        nb_ages = len(age_bins)
    else:
        nb_ages = 1
        age_bins = np.ones((1,))

    wts = np.zeros((age_bins.flatten().shape[0], 2))
    if use_bins:
        age_bin_edges = bin_centers_to_edges(age_bins)

    # get atlas predictions (for M only so far)
    if conditional:
        atlas_preds = []
        for si in [1, 2]:
            attributes = [np.stack([f, si]) for f in age_bins]  # only for M this time
            atlas_preds.append(plot_cond_atl(model_atlas_out, attributes,
                                             atlas_vol_k, tqdm=None, show=False))

    else:
        atlas_preds = [model_atlas_out]

    print('hi')

    # gather stats
    subj_attr = np.zeros((nb_subjects, 2))
    lab_stats = []
    atlas_segs = [[0] * nb_ages, [0] * nb_ages]
    for subji in tqdm(range(nb_subjects)):
        if conditional:
            sample = next(volseg_genobj.cond_atl_data(batch_size=1))  # get cond, atlas, [vol, seg]
            sample_for_flow = [sample[0], sample[1], sample[2][0]]

            # gather stats
            age, sex = sample[0].flatten()
            sex = np.array(sex).astype('int')
            subj_attr[subji, :] = [age, sex]

        else:
            sample = next(volseg_genobj.atl_data(batch_size=1))  # get atlas, [vol, seg]
            sample_for_flow = [sample[0], sample[1][0]]
            sex = 1

        # gather segs
        seg = sample[-1][-1]
        seg_prob = np.concatenate([seg == f for f in des_labels], -1)
        # seg_prob = np.concatenate([1-np.sum(seg_prob,-1,keepdims=True).clip(0,1), seg_prob], -1)
        # seg_prob = seg_prob / (np.maximum(seg_prob, -1, keepdims=1), 1e-7)

        # do
        if use_bins:
            x_idx = np.digitize(age, age_bin_edges, right=True) - 1  # put in bins

            if conditional and replace_age:
                sample_for_flow[0][0][0] = age_bins[x_idx]
                # sample_for_flow[0][0][1] = 1

            # get flow, propagate segs
            neg_flow = inv_flow_model.predict(sample_for_flow)
            all_seg_onehot, seg_counts = _core_warp(seg, seg_prob, neg_flow)
            lab_stats.append(seg_counts)

            # atlas_segs[x_idx] += all_seg_onehot
            if conditional:
                wt = np.exp(-decay**2 * np.square(age - age_bins[x_idx]))
            else:
                wt = 1
            atlas_segs[sex - 1][x_idx] += all_seg_onehot * wt
            wts[x_idx, sex - 1] += wt

        else:
            for x_idx, abin in enumerate(age_bins):

                if conditional and replace_age:
                    sample_for_flow[0][0][0] = age_bins[x_idx]
                    # sample_for_flow[0][0][1] = 1

                # get flow, propagate segs
                neg_flow = inv_flow_model.predict(sample_for_flow)
                all_seg_onehot, seg_counts = _core_warp(seg, seg_prob, neg_flow)
                lab_stats.append(seg_counts)

                # add tot *that* atlas_seg
                if conditional:
                    wt = np.exp(-decay**2 * np.square(age - abin))
                else:
                    wt = 1
                atlas_segs[sex - 1][x_idx] += all_seg_onehot * wt
                wts[x_idx, sex - 1] += wt

    for si in [0, 1]:
        atlas_segs[si] = [atlas_segs[si][f] / wts[f, si] for f in range(len(atlas_segs[si]))]

    if conditional:
        lab_stats = np.stack(lab_stats, 0)

    atlas_segs_final = atlas_segs
    if not return_seg_prob:
        atlas_segs_final = [np.argmax(f, -1).squeeze() for f in atlas_segs]

    return atlas_preds, atlas_segs_final, subj_attr, lab_stats


def visualize_vol_segs_overlap_3D(vols, segs, slice_nr, axis, thickness=3, rot=-1, flip=False,
                                  colors=None, width=17, view_labels_only=False, upsample_factor=1):

    nb_vols = len(vols)

    if colors is None:
        nb_labels = 256
        np.random.seed(0)
        colors = np.random.random((nb_labels, 3)) * 0.5 + 0.5
        colors[0, :] = [0, 0, 0]

    vols_slices = [np.take(f.squeeze(), slice_nr, axis) for f in vols]
    atlas_segs_slices = [np.take(f.squeeze(), slice_nr, axis) for f in segs]
    print(len(atlas_segs_slices))

    # upsample slice
    if upsample_factor > 1:
        vols_slices = [scipy.ndimage.interpolation.zoom(
            f, upsample_factor, order=1) for f in vols_slices]
        atlas_segs_slices = [scipy.ndimage.interpolation.zoom(
            f, upsample_factor, order=1) for f in atlas_segs_slices]

    # get seg_overlap
    olap_fn = lambda v, s: pynd.segutils.seg_overlap(
        v, s, thickness=thickness, cmap=colors, do_contour='inner')

    segolap_slices = [olap_fn(vols_slices[f].squeeze(
    ), atlas_segs_slices[f].astype('int')) for f in range(nb_vols)]
    segolap_slices = [np.rot90(f.clip(0, 1), rot) for f in segolap_slices]
    if flip:
        segolap_slices = [np.fliplr(f) for f in segolap_slices]
    segolap_slices_single = [np.concatenate(segolap_slices, 1)]
    ne.plot.slices(segolap_slices_single, cmaps=['gray'], width=width)

    if view_labels_only:
        segolap_slices = [np.concatenate([np.rot90(f, -1) for f in atlas_segs_slices], 1)]
        ne.plot.slices(segolap_slices, width=width)

    return segolap_slices_single


def plot_vol_seg_stats(atlas_segs_final, age_bins, lab_stats, subj_attr, des_label_names,
                       nb_x_bins=None, figsize=(17, 5), second_plot_only=False, ylim=None):
    """
    """

    nb_labels = len(des_label_names)
    assert len(atlas_segs_final[0]) == len(age_bins)
    assert len(atlas_segs_final[1]) == len(age_bins)

    if nb_x_bins is None:
        nb_x_bins = len(age_bins)

    # sort and bin subj_attr
    xi = np.argsort(subj_attr)
    subj_attr_sorted = np.array([subj_attr[f] for f in xi]).flatten()
    subj_attr_bins = np.linspace(subj_attr.min(), subj_attr.max(), nb_x_bins)

    # plot binned stats from subjects
    y = plt.figure(figsize=figsize)

    if not second_plot_only:
        plt.subplot(1, 2, 1)
        for i in range(nb_labels):
            these_stats = scipy.stats.binned_statistic(subj_attr_sorted, np.array(
                [lab_stats[f, i] for f in xi]), bins=subj_attr_bins, statistic='mean')
            plt.plot(bin_edges_to_center(subj_attr_bins), these_stats.statistic, '.-')
        plt.legend(des_label_names)
        plt.xlabel('age (binned)')
        plt.xticks(bin_edges_to_center(subj_attr_bins))
        plt.grid()
        plt.ylabel('volume')
        plt.title('stats averages from subjects')

        plt.subplot(1, 2, 2)

    y_st = [[[np.sum(f == g) for g in range(1, nb_labels + 1)]
             for f in atlas_segs_final[f]] for f in [0, 1]]
    y_st = np.array(y_st)

    plt.plot(age_bins, y_st[0, ...], '-')
    plt.plot(age_bins, y_st[1, ...], '.--')
    plt.legend(des_label_names)
    plt.xlabel('age (binned)')
    plt.xticks(age_bins)
    plt.grid()
    plt.ylabel('volume')
    plt.title('stats from averaged brains')

    if ylim is not None:
        plt.ylim(ylim)

    plt.show()
    return y


def plot_cond_reg():
    pass


class Generator():
    """
    Generators
    """

    def __init__(self, x_k=None, atlas_vol_k=None, y_k=None,
                 npz_varname=None, randomize=True, rand_seed=None,
                 onehot_classes=None, idxprob=None):
        self.x_k = x_k  # assumed k-size [nb_items, *vol_shape, 1]
        self.vol_shape = None
        if type(self.x_k) is np.ndarray:
            self.vol_shape = self.x_k.shape[1:-1]
            self.numel = self.x_k.shape[0]
        else:
            assert isinstance(self.x_k, (list, tuple)), "x_k should be numpy array or list/tuple"
            self.numel = len(self.x_k)
        self.atlas_k = atlas_vol_k
        self.y_k = y_k
        self.npz_varname = npz_varname
        self.randomize = randomize
        self.rand_seed = rand_seed
        if self.rand_seed is not None:
            np.random.seed(self.rand_seed)  # not sure how consistent this is in generators
        self.idx = [-1]
        self.onehot_classes = onehot_classes
        self.idxprob = idxprob  # prob to allot
        if self.idxprob is not None:
            assert len(self.idxprob) == self.numel

        if self.vol_shape is None:
            self._get_data([0])

    def atl_data(self, batch_size=32):
        """
        yield batches of (atlas, data)
        """

        atlas_k_bs = np.repeat(self.atlas_k, batch_size, axis=0)

        while 1:
            idx = self._get_next_idx(batch_size)
            x_sel = self._get_data(idx)

            yield (atlas_k_bs, x_sel)

    def mean_flow_x2(self, batch_size=32):
        """
        yield batches of [(atlas, data), (data, zeros, zeros, zeros)
        """
        zero_flow = np.zeros([batch_size, *self.vol_shape, len(self.vol_shape)])
        atl_data_gen = self.atl_data(batch_size=batch_size)

        while 1:
            a, x = next(atl_data_gen)
            yield ([a, x], [x, zero_flow, zero_flow, zero_flow])

    def mean_flow(self, batch_size=32):
        """
        yield batches of [(atlas, data), (data, zeros, zeros)
        """
        zero_flow = np.zeros([batch_size, *self.vol_shape, len(self.vol_shape)])
        atl_data_gen = self.atl_data(batch_size=batch_size)

        while 1:
            a, x = next(atl_data_gen)
            yield ([a, x], [x, zero_flow, zero_flow])

    def bidir_mean_flow(self, batch_size=32):
        """
        yield batches of [(atlas, data), (data, atlas, zeros, zeros)
        """
        zero_flow = np.zeros([batch_size, *self.vol_shape, len(self.vol_shape)])
        atl_data_gen = self.atl_data(batch_size=batch_size)

        while 1:
            a, x = next(atl_data_gen)
            yield ([a, x], [x, a, zero_flow, zero_flow])

    def cond(self, batch_size=32):
        """
        yield batches of (cond, atlas)

        where cond represent other data about the subject
        """

        while 1:
            idx = self._get_next_idx(batch_size)
            y_sel = self._get_att(idx)

            yield y_sel

    def cond_atl(self, batch_size=32):
        """
        yield batches of (cond, atlas)

        where cond represent other data about the subject
        """

        atlas_k_bs = np.repeat(self.atlas_k, batch_size, axis=0)

        while 1:
            idx = self._get_next_idx(batch_size)
            y_sel = self._get_att(idx)

            yield (y_sel, atlas_k_bs)

    def cond_atl_data(self, batch_size=32):
        """
        yield batches of (cond, atlas, data)

        where cond represent other data about the subject
        """

        atlas_k_bs = np.repeat(self.atlas_k, batch_size, axis=0)

        while 1:
            idx = self._get_next_idx(batch_size)
            x_sel = self._get_data(idx)
            y_sel = self._get_att(idx)

            yield (y_sel, atlas_k_bs, x_sel)

    def cond_mean_flow(self, batch_size=32):
        """
        yield batches of [(cond, atlas, data), (data, atlas, zeros, zeros)
        """
        zero_flow = np.zeros([batch_size, *self.vol_shape, len(self.vol_shape)])
        cond_atl_data_gen = self.cond_atl_data(batch_size=batch_size)

        while 1:
            c, a, x = next(cond_atl_data_gen)
            yield ([c, a, x], [x, zero_flow, zero_flow])

    def cond_mean_flow_x2(self, batch_size=32):
        """
        yield batches of [(cond, atlas, data), (data, atlas, zeros, zeros)
        """
        zero_flow = np.zeros([batch_size, *self.vol_shape, len(self.vol_shape)])
        cond_atl_data_gen = self.cond_atl_data(batch_size=batch_size)

        while 1:
            c, a, x = next(cond_atl_data_gen)
            yield ([c, a, x], [x, zero_flow, zero_flow, zero_flow])

    def cond_bidir_mean_flow(self, batch_size=32):
        """
        yield batches of [(cond, atlas, data), (data, atlas, zeros, zeros)
        """
        cond_atl_data_gen = self.cond_atl_data(batch_size=batch_size)

        while 1:
            c, a, x = next(cond_atl_data_gen)
            # need to build zero flow after first call in case vol_shape was None
            if self.vol_shape is not None:
                zero_flow = np.zeros([batch_size, *self.vol_shape, len(self.vol_shape)])

            yield ([c, a, x], [x, a, zero_flow, zero_flow])

    def cond_bidir_mean_flow_x2(self, batch_size=32):
        """
        yield batches of [(cond, atlas, data), (data, atlas, zeros, zeros, zeros)
        """
        cond_atl_data_gen = self.cond_atl_data(batch_size=batch_size)

        while 1:
            c, a, x = next(cond_atl_data_gen)
            # need to build zero flow after first call in case vol_shape was None
            if self.vol_shape is not None:
                zero_flow = np.zeros([batch_size, *self.vol_shape, len(self.vol_shape)])

            yield ([c, a, x], [x, a, zero_flow, zero_flow, zero_flow])

    def data(self, batch_size):
        while 1:
            idx = self._get_next_idx(batch_size)
            yield self._get_data(idx)

    def _get_next_idx(self, batch_size):
        if self.randomize:
            if self.idxprob is None:
                idx = np.random.randint(self.numel, size=batch_size)
            else:
                idx = np.random.choice(range(self.numel), size=batch_size, p=self.idxprob)
        else:
            idx = np.arange(self.idx[-1] + 1, self.idx[-1] + batch_size + 1)
            idx = np.mod(idx, self.numel)
        self.idx = idx
        return idx

    def _get_att(self, idx):
        if isinstance(self.y_k, (list, tuple, np.ndarray)):
            y_sel = self.y_k[idx]

            if self.onehot_classes is not None:
                y_sel = init_onehot(y_sel, self.onehot_classes)
        else:
            # assume it's dict
            if isinstance(self.x_k[0], (list, tuple)):
                y_sel = np.stack([self.y_k[self.x_k[i][0]] for i in idx], 0)
            else:
                y_sel = np.stack([self.y_k[self.x_k[i]] for i in idx], 0)
        return y_sel

    def _get_data(self, idx):
        if type(self.x_k) is np.ndarray:
            x = self.x_k[idx, ...]
        else:  # assume list of names or list of lists (e.g. if you have [vol, seg])
            if isinstance(self.x_k[0], (list, tuple)):
                batch_data = [[nes.data.load_image(f, npz_varname=self.npz_varname)
                               for f in self.x_k[i]] for i in idx]
                batch_data = list(map(list, zip(*batch_data)))  # transpose list
                x = [np.stack(f, 0)[..., np.newaxis] for f in batch_data]
                if self.vol_shape is None:
                    self.vol_shape = x[0].shape[1:-1]
            else:
                batch_data = [nes.data.load_image(
                    self.x_k[i], npz_varname=self.npz_varname) for i in idx]
                x = np.stack(batch_data, 0)[..., np.newaxis]
            if self.vol_shape is None:
                self.vol_shape = x.shape[1:-1]
        return x

    def one_shot_atlas_gen_MNIST(self, batch_size=32, des_labels=list(range(0, 7)),
                                 rotation_angle_range=None):
        x_train = self.x_k
        y_train = self.y_k
        z_im = np.zeros([batch_size, *x_train.shape[1:-1], 1])
        z_def = np.zeros([batch_size, *x_train.shape[1:-1], 2])

        while 1:
            idx = des_labels[np.random.randint(0, len(des_labels))]
            self.sel_digit = idx
            x = x_train[y_train == idx, ...]

            idx = np.random.randint(0, x.shape[0], size=(batch_size,))
            x = x[idx, ...]

            if rotation_angle_range is not None:
                angle = np.random.uniform(*rotation_angle_range)
                self.sel_angle = angle
                x = scipy.ndimage.rotate(x, angle=angle, axes=(1, 2), reshape=False)

            yield x, [z_im, z_def, z_def]


#########################################
# some specific helper functions
#########################################


def zoom_same_size(img, zoom, order=1):
    """
    zoom and crop/pad to the original image size.
    """

    # force final image size to be a multiple of 2 diffrerence
    im_shape = np.array(img.shape)
    new_shape = np.ceil(im_shape * zoom)
    diff = new_shape - im_shape
    for di, d in enumerate(diff):
        if d > 0 and (d % 2) == 1:
            new_shape[di] += 1
        if d < 0 and (-d % 2) == 1:
            new_shape[di] -= 1
    zoom = new_shape / im_shape

    # zoom
    im = scipy.ndimage.interpolation.zoom(img, zoom, order=order)

    # pad if necessarry
    pd = [np.maximum(img.shape[f] - im.shape[f], 0) for f in range(len(img.shape))]
    for p in pd:
        assert (p % 2) == 0, 'pad is not even. Something went wrong'

    if np.any(np.array(pd) > 0):
        pd = [(f // 2, f // 2) for f in pd]
        im = np.pad(im, pd, 'constant')

    # crop if necessary
    im = pynd.ndutils.volcrop(im, im_shape)
    assert np.all([np.array(im.shape) == im_shape]), 'failed sizes'
    return im


def init_onehot(elems, nb_classes=None):
    if nb_classes is None:
        nb_classes = np.max(elems) + 1
    assert np.max(elems) < nb_classes, 'onehot: max element is >= nb_classes'

    if not isinstance(elems, (list, tuple, np.ndarray)):
        elems = [elems]

    z = np.zeros((len(elems), nb_classes))
    for ei, e in enumerate(elems):
        z[ei, e] = 1
    return z


def bin_centers_to_edges(c, inf_edges=True):
    c = np.array(c).flatten()
    d = np.diff(c) / 2
    core = c[:-1] + d
    if inf_edges:
        edge_1 = -np.inf
        edge_2 = np.inf
    else:
        edge_1 = c[0] - d[0]
        edge_2 = c[-1] + d[-1]
    edge_1 = np.array(edge_1).reshape((1,))
    edge_2 = np.array(edge_2).reshape((1,))
    return np.concatenate([edge_1, core, edge_2])


def bin_edges_to_center(e):
    return e[:-1] + np.diff(e)


def tf_rot_to_affine_2d(a):
    """
    angle is a tf Tensor in the range -180, 180
    """
    a = -a / 180 * np.pi
    affine1 = tf.stack([tf.cos(a), -tf.sin(a), 0], 0)
    affine2 = tf.stack([tf.sin(a), tf.cos(a), 0], 0)
    affine3 = tf.stack([0., 0., 1], 0)
    affine = tf.stack([affine1, affine2, affine3], 0)
    return affine


def tf_rotate_2d(img, a, interp_method='linear'):
    shift = ne.utils.affine_to_shift(tf_rot_to_affine_2d(a), img.shape.as_list()[:-1])
    return ne.utils.transform(img, shift, interp_method='linear')


def k_rotate_2d(img, a, interp_method='linear'):
    """
    img is shape [bs, H, W, #]

    angle is a tf Tensor in the range -180, 180 of shape [bs]
    """
    print(img, a)
    fn = lambda x: tf_rotate_2d(x[0], x[1], interp_method)
    return tf.map_fn(fn, [img, a], dtype=tf.float32)


#########################################
# Losses
#########################################
ncc_loss = lambda yt, yp: 1 + losses.NCC(win=[]).loss(yt, yp)
mse_loss = lambda yt, yp: K.mean(K.square(yt - yp))
mse_ncc_mixed_loss = lambda yt, yp: 100 * mse_loss(yt, yp) + 1 * ncc_loss(yt, yp)
msmag_loss = lambda _, y_pred: K.mean(K.square(y_pred))
msmagvec_loss = lambda _, y_pred: K.mean(K.sum(K.square(y_pred), -1))
