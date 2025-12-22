"""
imputation project -- very experimental code below :(
"""

# built in
import os
import sys
import six
import types
import random

# third party imports
import numpy as np
import scipy
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf
# from keras_tqdm import TQDMNotebookCallback
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import tensorflow.keras.layers as KL
from tensorboard.backend.event_processing import event_accumulator
import tensorflow.keras.callbacks as keras_callbacks
import tensorflow.keras.utils as k_utils
import matplotlib.pylab as plt
from tqdm import tqdm
from scipy.ndimage.interpolation import rotate

# local import
import pystrum.pynd.ndutils as nd
import pystrum.pynd.segutils as su
import pystrum.pytools.timer as timer
import pystrum.pynd.patchlib


# TODO: make this into a Layer() ! The inner variables could be self . accessible...
def sparse_conv_block(data, mask, conv_layer, mask_conv_layer, core_name):
    '''
    sparse convolutional block.

    data is the data tensor
    mask is a binary tensor the same size as data

    steps:
    - set empty voxels in data using data *= mask
    - conv data and mask with the conv conv_layer
    - re-weight data
    - binarize mask
    '''

    # make sure the data is sparse according to the mask
    wt_data = keras.layers.Lambda(
        lambda x: x[0] * x[1], name='%s_pre_wmult' % core_name)([data, mask])

    # convolve data
    conv_data = conv_layer(wt_data)

    # convolve mask with all-ones.
    # necessary for normalization, etc.
    conv_mask = mask_conv_layer(mask)
    zero_mask = keras.layers.Lambda(lambda x: x * 0 + 1)(mask)
    conv_mask_allones = mask_conv_layer(zero_mask)  # all_ones mask to get the edge counts right.
    mask_conv_layer.trainable = False
    o = np.ones(mask_conv_layer.get_weights()[0].shape)
    mask_conv_layer.set_weights([o])

    # re-weight data (this is what makes the conv makes sense)
    data_norm = lambda x: x[0] * x[2] / (x[1] + 1e-5)  # updated 3/6/2019
    out_data = keras.layers.Lambda(data_norm, name='%s_norm_im' % core_name)([
        conv_data, conv_mask, conv_mask_allones])
    mask_norm = lambda x: tf.cast(x > 0, tf.float32)
    out_mask = keras.layers.Lambda(mask_norm, name='%s_norm_wt' % core_name)(conv_mask)

    return (out_data, out_mask)


def conv_block_W(data, mask, conv_layer, core_name):
    '''
    data is the data tensor
    mask is a binary tensor the same size as data
    '''

    # make sure the data is sparse according to the mask
    # wt_data = keras.layers.Lambda(lambda x: x[0] * x[1],
    #   name='%s_pre_wmult' % core_name)([data, mask])
    # wt_data = data * mask

    # get G
    if not hasattr(conv_layer, 'kernel'):
        _ = conv_layer(data)
    F = conv_layer.kernel  # k x k x ci x co
    F_flat = K.reshape(F, [-1, K.shape(F)[-1], 1])  # k*k*ci x co x 1

    F_flat = K.permute_dimensions(F_flat, [1, 0, 2])  # co x k*k*ci x 1
    FtF = K.batch_dot(K.permute_dimensions(F_flat, [0, 2, 1]), F_flat)  # co x 1 x 1

    G = F_flat / K.maximum(FtF, K.epsilon())  # co x k*k*ci x 1
    G = K.permute_dimensions(G, [1, 0, 2])  # k*k*ci x co x 1
    G = K.reshape(G, K.shape(F))  # k x k x ci x co

    # compute Go Io via conv --> N x length x width x co
    GoIo = K.conv2d(data * mask, G, strides=conv_layer.strides, padding=conv_layer.padding)

    # compute G**2 and convolve with weights --> N x length x width x co
    # /np.prod(K.shape(F)[:3])
    GoNorm2 = K.conv2d(mask, K.pow(G, 2), strides=conv_layer.strides, padding=conv_layer.padding)
    hack_m = K.conv2d(mask, G * 0 + 1, strides=conv_layer.strides, padding=conv_layer.padding)
    hack_maxm = K.cast(K.prod(K.shape(F)[:3]), tf.float32)

    #
    # Go = K.conv2d(mask, G, strides=conv_layer.strides, padding=conv_layer.padding)

    # return the division
    # 0.01 - 0.01 * hack_m / hack_maxm
    return GoIo / (GoNorm2 + 1e-4), GoNorm2, GoIo, G, (hack_m / hack_maxm)  #


def conv_enc(nb_features,
             input_shape,
             nb_levels,
             conv_size,
             name=None,
             prefix=None,
             feat_mult=1,
             pool_size=2,
             padding='same',
             activation='elu',
             layer_nb_feats=None,
             use_residuals=False,
             nb_conv_per_level=2,
             conv_dropout=0,
             double_last_conv_layer=False,
             batch_norm=None,
             nargout=1):
    """
    Sparse Fully Convolutional Encoder
    This is a fairly ugly copy from neurite.models, with changes appropriate for sparse convolutions
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # volume size data
    ndims = len(input_shape) - 1
    input_shape = tuple(input_shape)
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    # prepare layers
    convL = getattr(KL, 'Conv%dD' % ndims)
    conv_kwargs = {'padding': padding, 'activation': activation}
    maxpool = getattr(KL, 'MaxPooling%dD' % ndims)

    # first layer: input
    name = '%s_input' % prefix
    last_tensor = KL.Input(shape=input_shape, name=name)
    name = '%s_input_mask' % prefix
    mask_tensor = KL.Input(shape=input_shape, name=name)
    input_tensors = [last_tensor, mask_tensor]

    # down arm:
    # add nb_levels of conv + ReLu + conv + ReLu. Pool after each of first nb_levels - 1 layers
    lfidx = 0
    for level in range(nb_levels):
        lvl_first_tensor = last_tensor
        nb_lvl_feats = np.round(nb_features * feat_mult**level).astype(int)

        for conv in range(nb_conv_per_level):
            if layer_nb_feats is not None:
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += 1

            if conv == (nb_conv_per_level - 1) and level == (nb_levels - 1) \
                    and double_last_conv_layer:
                # do the secondary block
                name = '%s_conv_downarm_logvar_%d_%d' % (prefix, level, conv)

                im_conv = convL(nb_lvl_feats, conv_size, padding=padding,
                                name='%s_conv' % name, strides=strides)
                if conv_dropout > 0:
                    name = '%s_dropout_downarm_%d_%d' % (prefix, level, conv)
                    im_conv = KL.Dropout(conv_dropout)(im_conv)
                mask_conv = convL(1, conv_size, padding=padding, name='%s_wt' %
                                  name, use_bias=False, strides=strides)
                last_tensor_double, mask_tensor_double = sparse_conv_block(
                    last_tensor, mask_tensor, im_conv, mask_conv, name)

                print('cancelled act')
                # if conv < (nb_conv_per_level-1) or (not use_residuals):
                # name = '%s_conv_downarm_logvar_%d_%d_act' % (prefix, level, conv)
                # last_tensor_double = KL.Activation(activation, name=name)(last_tensor_double)

            strides = [1] * ndims
            if level < (nb_levels - 1) and (conv == nb_conv_per_level - 1):
                strides = [2] * ndims

            # do the convolution block
            name = '%s_conv_downarm_%d_%d' % (prefix, level, conv)

            im_conv = convL(nb_lvl_feats, conv_size, padding=padding,
                            name='%s_conv' % name, strides=strides)
            if conv_dropout > 0:
                name = '%s_dropout_downarm_%d_%d' % (prefix, level, conv)
                im_conv = KL.Dropout(conv_dropout)(im_conv)
            mask_conv = convL(1, conv_size, padding=padding, name='%s_wt' %
                              name, use_bias=False, strides=strides)
            last_tensor, mask_tensor = sparse_conv_block(
                last_tensor, mask_tensor, im_conv, mask_conv, name)

            if conv < (nb_conv_per_level - 1) or (not use_residuals):
                name = '%s_conv_downarm_%d_%d_act' % (prefix, level, conv)
                last_tensor = KL.Activation(activation, name=name)(last_tensor)
            # otherwise activation

        if use_residuals:
            print("residuals not implemented for sparse convolutions. Activating and Skipping",
                  file=sys.stderr)
            # TWO problems: how to add/subtract, and best way to do the striding? another conv??

            # convarm_layer = last_tensor

            # # the "add" layer is the original input
            # # However, it may not have the right number of features to be added
            # nb_feats_in = lvl_first_tensor.get_shape()[-1]
            # nb_feats_out = convarm_layer.get_shape()[-1]
            # add_layer = lvl_first_tensor
            # if nb_feats_in > 1 and nb_feats_out > 1 and (nb_feats_in != nb_feats_out):
            #     name = '%s_expand_down_merge_%d' % (prefix, level)
            #     last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)
            #       (lvl_first_tensor)
            #     add_layer = last_tensor

            # name = '%s_res_down_merge_%d' % (prefix, level)
            # last_tensor = KL.add([add_layer, convarm_layer], name=name)

            name = '%s_conv_downarm_%d_%d_act' % (prefix, level, conv)
            last_tensor = KL.Activation(activation, name=name)(last_tensor)

            # if conv == (nb_conv_per_level-1) and level == (nb_levels-1)
            #   and double_last_conv_layer:
            #     name = '%s_conv_downarm_logvar_%d_%d_act' % (prefix, level, conv)
            #     last_tensor_double = KL.Activation(activation, name=name)(last_tensor_double)

        if batch_norm is not None:
            name = '%s_bn_down_%d' % (prefix, level)
            last_tensor = KL.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

        # max pool if we're not at the last level
        # if level < (nb_levels - 1):
            # name = '%s_maxpool_%d' % (prefix, level)
            # last_tensor = maxpool(pool_size=pool_size, name=name, padding=padding)(last_tensor)

    # create the model and return
    model = keras.models.Model(inputs=input_tensors, outputs=[last_tensor], name=model_name)
    if nargout == 1:
        return model
    elif nargout == 3:
        return (model, last_tensor, mask_tensor)
    else:
        assert nargout == 5
        return (model, last_tensor, mask_tensor, last_tensor_double, mask_tensor_double)


def load_data(dataset_name, sparsity, load_path=None, rand_seed=None, tqdm=tqdm,
              do_rot=False,
              sp_pattern='uniform',
              do_linear_interp=True,
              nb_copy=0,
              y_cond_equal=None,
              extract_random_patch_sizes=None,
              maskthr=None,
              do_shifts=None,
              do_resize=None,
              vols_name='vols',
              line_rot_max_angle=180):

    if rand_seed is not None:
        np.random.seed(rand_seed)

    data_types = ['train', 'test', 'val']
    folders = ['train', 'test', 'validate']
    # data_types = ['test']
    # folders = ['test']
    x = {}
    mx = {}
    y = {}

    if dataset_name in ['boston', 'boston-top']:
        nb_maxs = [20000, 5000, 5000]
        for di, dt in enumerate(data_types):
            x[dt] = load_jpg_data(os.path.join(load_path, folders[di]),
                                  nb_max=nb_maxs[di], tqdm=tqdm)
            y[dt] = np.ones(x[dt].shape[0])

            if dataset_name == 'boston-top':
                x[dt] = x[dt][:, :64, :]

    if dataset_name in ['omniglot']:
        # /data/ddmg/voxelmorph/data/omniglot/proc/nips2018_nostructure
        nb_maxs = [20000, 5000, 5000]
        for di, dt in enumerate(data_types):
            x[dt] = load_jpg_data(os.path.join(load_path, folders[di]),
                                  nb_max=nb_maxs[di], tqdm=tqdm, ext='png', do_crop=False)
            y[dt] = np.ones(x[dt].shape[0])
            x[dt] = (1 - x[dt][:, 4:-5, 4:-5]) * 255

    elif dataset_name in ['brain-crop', 'brain-crop-tiny']:
        for di, dt in enumerate(data_types):
            x[dt] = load_npz_data(os.path.join(load_path, '%s/%s' %
                                               (folders[di], vols_name)), tqdm=tqdm)
            x[dt] = x[dt].squeeze()
            x[dt] = x[dt] * 255
            y[dt] = np.ones(x[dt].shape[0])

            if dataset_name == 'brain-crop-tiny':
                x[dt] = x[dt][:, -64:, 80:144]

    elif dataset_name in ['brain-crop-real-iso']:
        for di, dt in enumerate(data_types):
            qpath = os.path.join(load_path, '%s/slices' % folders[di])
            subjects = os.listdir(qpath)
            isovols = []
            for subj in tqdm(subjects):
                isovols += [np.load(
                    os.path.join(qpath, subj,
                                 '%s_iso_2_ds5_us5_size_reg.npz' % subj))['vol_data']]

            x[dt] = np.stack(isovols, 0) * 255
            y[dt] = np.ones(x[dt].shape[0])

    elif dataset_name in ['brain-real-slice']:
        # prepped for /home/gid-dalcaav/projects/neurite/data/adni_sparse_synth/proc/reg/train/vols/
        for di, dt in enumerate(data_types):
            qpath = os.path.join(load_path, '%s/slices' % folders[di])
            subjects = os.listdir(qpath)

            vols = []
            masks = []
            isovols = []
            for subj in tqdm(subjects):
                vols += [np.load(os.path.join(qpath, subj, '%s_ds5_us5_reg.npz' %
                                              subj))['vol_data']]
                masks += [np.load(os.path.join(qpath, subj,
                                               '%s_ds5_us5_dsmask_reg.npz' % subj))['vol_data']]
                isovols += [np.load(
                            os.path.join(qpath, subj,
                                         '%s_iso_2_ds5_us5_size_reg.npz' % subj))['vol_data']]

            x[dt] = np.stack(vols, 0) * 255
            x[dt + '_orig'] = np.stack(isovols, 0) * 255
            y[dt] = np.ones(x[dt].shape[0])

            mx[dt] = np.stack(masks, 0)
            mx[dt + '_orig'] = np.copy(mx[dt])
            if maskthr is not None:
                mx[dt] = mx[dt] > maskthr

    elif dataset_name == 'mnist':
        (x['train'], y['train']), (x['test'], y['test']) = mnist.load_data()

    elif dataset_name == 'cifar10':
        (x['train'], y['train']), (x['test'], y['test']) = cifar10.load_data()
        x['train'] = rgb2gray(x['train'])
        x['test'] = rgb2gray(x['test'])

    elif dataset_name == 'fashion_mnist':
        (x['train'], y['train']), (x['test'], y['test']) = fashion_mnist.load_data()

    else:
        assert False, 'unknown dataset'

    if dataset_name in ['mnist', 'fashion_mnist', 'cifar10']:
        k = (x['train'].shape[0] * 5) // 6
        x['val'] = x['train'][k:, :]
        y['val'] = y['train'][k:]
        x['train'] = x['train'][:k, :]
        y['train'] = y['train'][:k]

    if y_cond_equal is not None:
        for dt in data_types:
            x[dt] = x[dt][y[dt] == y_cond_equal, :]
            y[dt] = y[dt][y[dt] == y_cond_equal]

    # copy datasets. This is useful to do if, for example,
    # rotating data at random (as done below) and we want more data
    if nb_copy > 0:
        for dt in tqdm(data_types, desc='copying'):
            sh = np.ones(x[dt].ndim).astype('int')
            sh[0] = nb_copy
            x[dt] = np.tile(x[dt], sh)
            y[dt] = np.tile(y[dt], (sh[0], ))
            if dt + '_orig' in x:
                x[dt + '_orig'] = np.tile(x[dt + '_orig'], sh)
            if dt in mx:
                mx[dt] = np.tile(mx[dt], sh)
            if dt + '_orig' in mx:
                mx[dt + '_orig'] = np.tile(mx[dt + '_orig'], sh)

    # roate x if doing rotations
    if do_rot:
        for dt in data_types:
            x[dt] = np.copy(x[dt])
            for si in tqdm(range(x[dt].shape[0]), desc='rotating'):
                r = np.random.randint(0, 360)
                x[dt][si, :] = rotate(x[dt][si, :], r, reshape=False)

    if do_resize:
        for dt in data_types:
            xdt = []
            for si in tqdm(range(x[dt].shape[0]), desc='resizing'):
                xdt += [scipy.misc.imresize(x[dt][si, :], do_resize)]
            x[dt] = np.stack(xdt, 0)

    if do_shifts is not None and dataset_name in ['brain-real-slice']:
        for dt in data_types:
            xdt = []
            xdtorig = []
            mxdt = []
            mxdto = []
            for i in tqdm(range(x[dt].shape[0]), desc="shifting"):
                sx = np.random.randint(do_shifts[0][0], do_shifts[0][1])
                sy = np.random.randint(do_shifts[1][0], do_shifts[1][1])
                startx = 48 + sx
                starty = 63 + sy
                endx = 48 + 160 + sx
                endy = 63 + 144 + sy

                xdt += [x[dt][i, startx:endx, starty:endy]]
                xdtorig += [x[dt + '_orig'][i, startx:endx, starty:endy]]
                mxdt += [mx[dt][i, startx:endx, starty:endy]]
                if dt + '_orig' in mx:
                    mxdto += [mx[dt + '_orig'][i, startx:endx, starty:endy]]
            x[dt] = np.stack(xdt, 0)
            x[dt + '_orig'] = np.stack(xdtorig, 0)
            mx[dt] = np.stack(mxdt, 0)
            if dt + '_orig' in mx:
                mx[dt + '_orig'] = np.stack(mxdto, 0)

    elif dataset_name in ['brain-real-slice', 'brain-crop-real-iso']:
        for dt in data_types:
            x[dt] = x[dt][:, 48:48 + 160, 63:63 + 144]
            if dt + '_orig' in x:
                x[dt + '_orig'] = x[dt + '_orig'][:, 48:48 + 160, 63:63 + 144]
            if dt in mx:
                mx[dt] = mx[dt][:, 48:48 + 160, 63:63 + 144]
            if dt + '_orig' in mx:
                mx[dt + '_orig'] = mx[dt + '_orig'][:, 48:48 + 160, 63:63 + 144]

    # get sparsity pattern
    if sp_pattern == 'line-rot':
        line_density = np.round(1 / (1 - sparsity)).astype(int)
        for dt in data_types:
            mx[dt] = np.zeros(x[dt].shape)

            for si in tqdm(range(x[dt].shape[0]), desc='rotating-line'):
                ri = np.random.randint(0, line_density)
                mx[dt][si, :, ri::line_density] = 1

                r = np.random.randint(-line_rot_max_angle, line_rot_max_angle)
                mx[dt][si, :] = np.round(rotate(mx[dt][si, :], r, reshape=False))

    elif sp_pattern == 'uniform':
        for dt in data_types:
            mx[dt] = (np.random.random(x[dt].shape) > sparsity).astype('float32')
    else:
        for dt in data_types:
            if dt not in mx:
                mx[dt] = np.ones(x[dt].shape)

    # cleanup empty weights data
    for dt in data_types:
        good_idx = np.where(mx[dt].sum((1, 2)) > 5)[0]
        x[dt] = x[dt][good_idx, :]
        if dt + '_orig' in x:
            x[dt + '_orig'] = x[dt + '_orig'][good_idx, :]
        mx[dt] = mx[dt][good_idx, :]
        if dt + '_orig' in mx:
            mx[dt + '_orig'] = mx[dt + '_orig'][good_idx, :]
        y[dt] = y[dt][good_idx]

    # clean-up and sparsify
    for dt in data_types:
        x[dt] = np.expand_dims(x[dt], -1) / 255
        mx[dt] = np.expand_dims(mx[dt], -1)
        if dt + '_orig' in mx:
            mx[dt + '_orig'] = np.expand_dims(mx[dt + '_orig'], -1)
        if not dt + '_orig' in x:
            x[dt + '_orig'] = x[dt]
        else:
            x[dt + '_orig'] = np.expand_dims(x[dt + '_orig'], -1) / 255
        if sp_pattern is not None:
            x[dt] = x[dt] * mx[dt]

    if do_linear_interp:
        griddata = scipy.interpolate.griddata

        for data_t in data_types:
            dtname = '%s-linear-interp' % data_t
            x[dtname] = np.copy(x[data_t])

            for did in tqdm(range(x[data_t].shape[0]), desc='linear interp on %s' % data_t):

                im_val = x[data_t][did, :, :, 0]
                m_val = mx[data_t][did, :, :, 0]

                # do the linear interpolation
                if sparsity > 0:
                    known_idx = np.stack(np.where(m_val), 1)
                    known_values = im_val[known_idx[:, 0], known_idx[:, 1]]
                    unknown_idx = np.stack(np.where(np.logical_not(m_val)), 1)
                    imputed_values = griddata(known_idx, known_values, unknown_idx, fill_value=0)

                    x[dtname][did, unknown_idx[:, 0], unknown_idx[:, 1], 0] = imputed_values

    # extract some patches
    # using patchlib extract some patches
    if extract_random_patch_sizes is not None:
        nb_patch_copy = extract_random_patch_sizes[0]
        ps = extract_random_patch_sizes[1:]

        newx = {}
        for dt in x.keys():
            newx[dt] = np.zeros((x[dt].shape[0] * nb_patch_copy, *ps, 1))
            x_gen = patchlib.patch_gen(x[dt], [1, *ps, 1], rand=True, rand_seed=0)
            for si in tqdm(range(x[dt].shape[0] * nb_patch_copy), desc='patching x'):
                newx[dt][si, :] = next(x_gen)

            y[dt] = np.ones(newx[dt].shape[0])

        x = newx

        newmx = {}
        for dt in mx.keys():
            newmx[dt] = np.zeros((mx[dt].shape[0] * nb_patch_copy, *ps, 1))
            mx_gen = patchlib.patch_gen(mx[dt], [1, *ps, 1], rand=True, rand_seed=0)
            for si in tqdm(range(mx[dt].shape[0] * nb_patch_copy), desc='patching mx'):
                newmx[dt][si, :] = next(mx_gen)
        mx = newmx

    return (x, mx, y)


class Linear_EM():
    """
    keras/tf function

    OLD VIS CODE
    #     ry = em.recon_y()
    #     ne.plt.slices([f.reshape(run.patch_size) for f in [p[0], ry[0], ry[1]]])
    # reload(sb_imp)
    # r = 1
    # ys = x['val'][1, np.newaxis, :].flatten().reshape((1,-1))
    # ms = mx['val'][1, np.newaxis, :].flatten().reshape((1,-1))
    # ry = em.recon_y(ys=ys, ms=ms)
    # ne.plt.slices([f.reshape(run.patch_size) for f in [p[0], ry[0,:]]])
    """

    def __init__(self, y, m, batch_size, enc_size):
        self.y = K.variable(y)
        y_shape_orig = self.y.get_shape().as_list()
        self.m = K.variable(m)
        self.batch_size = batch_size

        self.y_shape = self.y.get_shape().as_list()

        self.y = K.batch_flatten(self.y)  # nb x D
        self.m = K.batch_flatten(self.m)  # nb x D

        self.D = np.prod(y_shape_orig[1:])
        self.d = enc_size

        # initialize params (random?)
        self.params = {}
        self.params['W'] = K.variable(K.random_normal([1, self.D, self.d]))  # 1 x D x d
        self.params['mu'] = K.variable(K.random_normal((1, self.D)))  # 1 x D
        self.params['s2'] = K.variable(K.random_normal((1, ), stddev=0.001))

        self.x = None  # nb x d
        self.s = None  # nb x d x d

        self.ys_one = K.zeros((1, self.D))
        self.ms_one = K.zeros((1, self.D))
        self.xs_one = self.y_to_x(self.ys_one, self.ms_one)

    def fit(self, max_iter=100, do_y_proj=False, tqdm=tqdm):
        x, s = self.estep()  # e-step graph
        mu, W, s2 = self.mstep(x, s)  # m-step graph
        params = [mu, W, s2]

        for i in tqdm(range(max_iter), desc='fit'):
            # do one em iter
            sess = K.get_session()
            params_eval = sess.run(params)
            params_eval[-1] = np.array([params_eval[-1]])
            self.params['mu'].load(params_eval[0], K.get_session())
            self.params['W'].load(params_eval[1], K.get_session())
            self.params['s2'].load(params_eval[2], K.get_session())

            # should do python loop
            # print(self.loss(tqdm=tqdm))
        self.x = x
        self.s = s

        return params_eval

    def recon_y_from_tf(self, ys, ms):
        # all-tf

        x = self.y_to_x(ys, ms)

        w = K.permute_dimensions(self.params['W'][0, :, :], [1, 0])  # d x D
        recon_y = self.params['mu'] + K.dot(x, w)  # N x D
        return recon_y

    def recon_y(self, ys=None, ms=None):

        if ys is not None:
            if isinstance(ys, tf.Variable):
                ys = K.eval(ys)
            if isinstance(ms, tf.Variable):
                ms = K.eval(ms)

            sess = K.get_session()
            self.ys_one.load(ys, sess)
            self.ms_one.load(ms, sess)
            x = K.variable(sess.run(self.xs_one))

        else:
            x = self.x

        w = K.permute_dimensions(self.params['W'][0, :, :], [1, 0])  # d x D
        recon_y = self.params['mu'] + K.dot(x, w)  # N x D
        return K.eval(recon_y)

    def y_to_x(self, ys, ms):

        # e-step:
        nb = 1
        eye = K.expand_dims(K.eye(self.d), 0)

        # take out batch
        # ys = ys  # 1 x D x 1
        # ms = ms  # 1 x D x 1

        W = self.params['W']  # 1 x D x d

        # prep quantities
        W0 = K.expand_dims(ms, -1) * W  # 1 x D x d
        W0T = K.permute_dimensions(W0, [0, 2, 1])  # 1 x d x D
        seye = self.params['s2'] * eye  # 1 x d x d
        pre = tf.matrix_inverse(K.batch_dot(W0T, W0) + seye)  # 1 x d x d
        m = K.batch_dot(W0T, ys)   # 1 x d
        x = K.batch_dot(pre, m)  # 1 x d

        return x

    def estep(self):
        """
        UNTESTED
        """

        # e-step:
        nb = self.y_shape[0]
        eye = K.expand_dims(K.eye(self.d), 0)

        # take out batch
        ys = self.y  # batch_size x D x 1
        ms = self.m  # batch_size x D x 1

        W = K.repeat_elements(self.params['W'], nb, 0)  # batch_size x D x d

        # prep quantities
        W0 = K.expand_dims(ms, -1) * W  # batch_size x D x d
        W0T = K.permute_dimensions(W0, [0, 2, 1])  # batch_size x d x D
        seye = self.params['s2'] * eye  # batch_size x d x d
        pre = tf.matrix_inverse(K.batch_dot(W0T, W0) + seye)  # batch_size x d x d
        m = K.batch_dot(W0T, ys)   # batch_size x d
        x = K.batch_dot(pre, m)  # batch_size x d

        # S
        s = self.params['s2'] * pre  # batch_size x d x d

        return (x, s)

    def batch_estep(self):
        nb = self.y_shape[0]
        eye = K.expand_dims(K.eye(self.d), 0)

        # e-step:
        this_x = []
        this_s = []

        for i in range(0, nb, self.batch_size):

            j = np.minimum(i + self.batch_size, nb)

            # take out batch
            ys = self.y[i:j, :]  # batch_size x D x 1
            ms = self.m[i:j, :]  # batch_size x D x 1

            W = K.repeat_elements(self.params['W'], j - i, 0)  # batch_size x D x d

            # prep quantities
            W0 = K.expand_dims(ms, -1) * W  # batch_size x D x d
            W0T = K.permute_dimensions(W0, [0, 2, 1])  # batch_size x d x D
            seye = self.params['s2'] * eye  # batch_size x d x d
            pre = tf.matrix_inverse(K.batch_dot(W0T, W0) + seye)  # batch_size x d x d
            m = K.batch_dot(W0T, ys)   # batch_size x d
            this_x += [K.batch_dot(pre, m)]  # batch_size x d

            # S
            this_s += [self.params['s2'] * pre]  # batch_size x d x d

        x = tf.concat(this_x, 0)
        s = tf.concat(this_s, 0)
        return (x, s)

    def mstep(self, x, s):
        # numbers of obs for each voxel j
        nbj = K.sum(self.m, 0)  # D
        onbj = K.expand_dims(1 / nbj, -1)  # D x 1

        # prep some variables
        x1 = K.expand_dims(x, -1)  # N x d x 1
        xt = K.permute_dimensions(x1, [0, 2, 1])  # N x 1 x d
        xxt = K.batch_dot(x1, xt)  # N x d x d --- note this is large

        # comput bj, Aj
        mt = K.transpose(self.m)  # D x N
        print(mt, x)
        B = onbj * K.dot(mt, x)  # D x d
        print(B)
        xxtms = s + xxt
        xxtSm = K.transpose(K.dot(K.permute_dimensions(xxtms, [1, 2, 0]), self.m))  # D x d x d
        A = K.expand_dims(onbj, -1) * xxtSm  # D x d x d
        Ainv = tf.matrix_inverse(A)  # D x d x d
        Ainvb = K.batch_dot(Ainv, B)  # D x d

        # compute mu
        xAinvb = K.dot(x, K.transpose(Ainvb))
        omAinvb = self.m * (1 - xAinvb)  # N x D
        mu = K.sum(omAinvb * self.y, 0, keepdims=True) / K.sum(omAinvb, 0, keepdims=True)  # 1 x D

        # W
        my_cen = self.m * (self.y - mu)  # N x D
        # N x d x D -- this is really large!
        xAinv = K.dot(x, K.permute_dimensions(Ainv, [1, 2, 0]))
        m_xAinv = K.expand_dims(K.transpose(onbj), 0) * \
            K.expand_dims(my_cen, 1) * xAinv  # N x d x D
        W = K.permute_dimensions(K.sum(m_xAinv, 0, keepdims=True), [0, 2, 1])  # 1 x D x d

        # s2
        wt = K.transpose(W[0, :, :])  # d x D
        wx = self.m * K.dot(x, wt)  # N x D
        term1 = K.sum((my_cen - wx) ** 2)

        # sd = K.dot(s, wt) # N x d x D
        # sdp = K.permute_dimensions(sd, [0, 2, 1])
        # term2all = K.dot(sdp,  wt)  # N x D x D
        # term2 = K.sum(tf.trace(term2all), 0) # 1
        # this might be faster and less memory intensive:
        wwt = K.dot(wt, K.transpose(wt))
        term2 = K.sum(tf.trace(K.dot(s, wwt)))
        s2 = (term1 + term2) / K.sum(self.m)

        return (mu, W, s2)

        # go through each voxel in D
        # for j in tqdm(range(self.D), desc='mstep'):
        #     onbj_j = onbj[j]

        #     mj = self.m[:, j]  # N
        #     yj = self.y[:, j]  # N

        # bj = onbj_j * K.sum(mj * x, 0)  # d
        # Aj = onbj_j * K.sum(mj * xxt_s, 0)  # d x d
        # Ajinv = tf.matrix_inverse(Aj)  # d x d
        # bj = K.expand_dims(bj, -1)  # d x 1

        # update mu
        # Ainvb = K.dot(Ajinv, bj)  # d x 1
        # para = 1 - K.dot(x,  Ainvb)  # N x 1
        # mu += [K.sum(mj * (para * yj), 0) / K.sum(mj * para, 0)]

        # update W
        # yjm = yj - mu[-1]  # N x 1
        # xAinv = tf.matmul(x, Ajinv)  # N x d
        # W += [onbj_j * K.sum(mj * yjm * xAinv, 0, keepdims=True)]  # 1 x d

        # update s2
        # this is expensive, so we will do it outside of the loop
        # w = K.permute_dimensions(W[-1], [1, 0])  # d x 1
        # wx = tf.matmul(x, w)  # N x 1
        # term1 = (yjm - wx) ** 2 # N x 1
        # term2 = K.dot(K.dot(s, w)[:,:,0],  w)  # N x 1

        # s_top += K.sum(term1 + term2, 0)  # 1
        # s_bot += K.sum(mj)  # 1

    def loss(self, tqdm=tqdm):
        nb = self.y_shape[0]

        # take out mean
        mu = self.params['mu'][0, :]

        # prep function
        MNFC = tf.contrib.distributions.MultivariateNormalFullCovariance

        s = 0
        for i in tqdm(range(nb)):
            mi = self.m[i, :]  # D

            # prep mean and cov
            muo = mu * mi  # D
            W0 = K.expand_dims(mi, -1) * self.params['W'][0, :, :]
            W0T = K.permute_dimensions(W0, [1, 0])

            C = K.dot(W0, W0T) + self.params['s2'] * K.eye(self.D)

            mvn = MNFC(loc=muo, covariance_matrix=C)
            s += mvn.log_prob(self.y[i, :] * mi)

            if i > 10:
                print(K.eval(s))

        return s


# some silly little specific help tools
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def load_jpg_data(path, nb_max=None, tqdm=tqdm, ext='jpg', do_crop=True):
    files = [f for f in os.listdir(path) if f.endswith(ext)]

    ims = []
    for file in tqdm(files):
        im = plt.imread(os.path.join(path, file))
        if len(im.shape) > 2 and im.shape[2] > 1:
            im = rgb2gray(im.astype(float))
        if do_crop:
            im = im[3:-3, 4:-4]
        ims = [*ims, np.reshape(im, [1, *im.shape])]

        if nb_max is not None and nb_max <= len(ims):
            break

    ims = np.concatenate(ims, 0)
    return ims


def load_npz_data(path, tqdm=tqdm, subj_folders=False, ext='npz'):
    files = [f for f in os.listdir(path) if f.endswith(ext)]

    ims = []
    for file in tqdm(files):
        im = np.load(os.path.join(path, file))['vol_data']
        ims = [*ims, np.reshape(im, [1, *im.shape])]
    ims = np.concatenate(ims, 0)
    return ims


class PrintSlice2D(keras.callbacks.Callback):

    def __init__(self, model=None, period=10, input_vol=None, input_mask=None, x_orig=None,
                 x_other=None, vis_idx=1, clip=None):
        self.period = period
        self.started = False
        self.input_vol = input_vol
        self.input_mask = input_mask
        self.x_orig = x_orig
        self.x_other = x_other
        self.vis_idx = vis_idx
        self.clip = None
        self.this_model = model

    def on_train_begin(self, logs={}):
        # self.losses = []

        self.epochs = []
        self.l2_data_val = []
        self.l2_data_train = []

        if self.this_model is not None:
            self.model = self.this_model

#         self.fig = plt.figure()

    def on_epoch_end(self, epoch, logs={}):
        if not self.started:
            self.started = True
            self.first_epoch = epoch

        if np.mod(epoch - self.first_epoch, self.period) == 0:

            # visualize prediction
            sample_model(self.model, self.input_vol, self.input_mask, x_orig=self.x_orig,
                         nb_examples=1, idx=self.vis_idx,
                         do_mix=True, clip=self.clip)

            # self.epochs.append(epoch)
            # df = np.square(self.model.predict([x['val'][0:1000,:], mx['val'][0:1000,:]])
            #   - x['val_orig'][0:1000,:])
            # self.l2_data_val.append(np.mean(df))

            # df = np.square(self.model.predict([x['train'][0:1000,:], mx['train'][0:1000,:]])
            #   - x['train_orig'][0:1000,:])
            # self.l2_data_train.append(np.mean(df))

            # plt.figure(figsize=(5,3))
            # plt.plot(self.epochs, self.l2_data_val, '-', self.epochs, self.l2_data_train, '-')
            # plt.legend(['val', 'train'])
            # plt.ylabel('MSE of all data')
            # plt.xlabel('epoch')
            # plt.show()


class PrintTB(keras.callbacks.Callback):

    def __init__(self, log_dir, period=20, losses=['loss'], **kwargs):
        self.log_dir = log_dir
        self.period = period
        self.losses = losses
        self.kwargs = kwargs
        self.skipped_first = False

    def on_epoch_end(self, epoch, logs={}):
        if self.skipped_first and np.mod(epoch, self.period) == 0:
            ne.utils.tensorboard.plot_loss_values(self.log_dir, self.losses, **self.kwargs)
        if not self.skipped_first:
            self.skipped_first = True

    def on_train_begin(self, logs={}):
        print


def callbacks(run_dir,
              model_names=[''],
              at_batch_end=None,
              log_prefix='log',
              gpus=1,
              cb_verbose=True,
              tb_print_period=10,
              format='list'):
    """
    usual callbacks for segmentation
    """

    all_cb = {}

    for model_name in model_names:

        callbacks = {}

        # model saving
        hdf_dir = os.path.join(run_dir, 'hdf5')
        if not os.path.isdir(hdf_dir):
            os.mkdir(hdf_dir)
        filename = os.path.join(hdf_dir, model_name + 'model.{epoch:02d}.hdf5')
        if gpus == 1:
            callbacks['save'] = ne.callbacks.ModelCheckpoint(filename,
                                                             monitor='val_loss',  # is this needed?
                                                             verbose=cb_verbose,
                                                             save_best_only=False,
                                                             save_weights_only=False,
                                                             at_batch_end=at_batch_end,
                                                             mode='auto',
                                                             period=1)
        else:
            callbacks['save'] = ne.callbacks.ModelCheckpointParallel(filename,
                                                                     monitor='val_loss',  # needed?
                                                                     verbose=cb_verbose,
                                                                     save_best_only=False,
                                                                     save_weights_only=False,
                                                                     at_batch_end=at_batch_end,
                                                                     mode='auto',
                                                                     period=1)

        # weight checking.
        callbacks['weight_checking'] = ne.callbacks.ModelWeightCheck(at_batch_end=at_batch_end)

        # tensorboard
        log_dir = os.path.join(run_dir, log_prefix, model_name)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        callbacks['tensorboard'] = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                               histogram_freq=0,
                                                               write_graph=True,
                                                               write_images=True)

        # tqdm
        callbacks['tqdm'] = None  # TQDMNotebookCallback()

        callbacks['tensorboard-print'] = \
            PrintTB(log_dir,
                    period=tb_print_period,
                    losses=['loss', 'val_loss', 'mse_metric', 'val_mse_metric'],
                    ylim_last_steps=100)

        if format == 'list':
            callbacks = list(callbacks.values())

        all_cb[model_name] = callbacks

    # return the dictionary of callbacks
    return all_cb


def mse_metric(input_true_layer):

    def mse_metric(_, y_pred):
        return keras.losses.mean_squared_error(input_true_layer, y_pred)

    return mse_metric


def get_data_example(x, mx, x_orig=None, idx=None, ret_idx=False, nb_samples=1, rand_seed=None):

    if rand_seed is not None:
        np.random.seed(rand_seed)

    if idx is None:
        idx = np.random.randint(0, x.shape[0], nb_samples)[0]

    if nb_samples == 1:
        im_val = x[idx, np.newaxis, :]
        m_val = mx[idx, np.newaxis, :]
    else:
        im_val = x[idx, :]
        m_val = mx[idx, :]

    if x_orig is not None:
        if nb_samples == 1:
            im_val_orig = x_orig[idx, np.newaxis, :]
        else:
            im_val_orig = x_orig[idx, :]
        ret = (im_val, m_val, im_val_orig)
    else:
        ret = (im_val, m_val)

    if ret_idx:
        ret += (idx, )

    return ret


def gen_from_data(x, mx, idx=None, do_rand=True, rand_seed=0):

    if do_rand:
        np.random.seed(rand_seed)

    did = -1
    while True:
        if idx is None and do_rand:
            did = np.random.randint(0, x.shape[0], 1)[0]
        elif idx is None:
            did = did + 1
        else:
            did = idx

        yield get_data_example(x, mx, idx=did)


def sample_model(model, x, mx, x_orig=None, x_others=None, idx=None,
                 do_mix=True, nb_examples=1, y=None, nb_vae_samples=1, clip=None):

    for _ in range(nb_examples):
        data = get_data_example(x, mx, x_orig=x_orig, idx=idx, ret_idx=True)
        sidx = data[-1]

        # initial images
        vols, titles = _init_vols_titles(data, sidx, y, x_orig, x_others)

        # add model result
        in_data = list(data[0:2])
        if nb_vae_samples > 1:
            in_data = [f.repeat(nb_vae_samples, 0) for f in in_data]
        res = model.predict(in_data)
        if isinstance(res, (list, tuple)):
            res = res[0]
        if nb_vae_samples > 1:
            res = np.mean(res, 0, keepdims=True)

        if clip is not None:
            res = np.clip(res, clip[0], clip[1])

        vols += [res]
        titles += ['recon %3.4f %3.4f' %
                   (_mse(res, data[2]), _mse(res * data[1], data[1] * data[0]) / data[1].mean())]

        if do_mix:
            vols += [data[0] * data[1] + res * (1 - data[1])]
            titles += ['mix %3.4f %3.4f' %
                       (_mse(vols[-1], data[2]),
                        _mse(vols[-1] * data[1], data[1] * data[0]) / data[1].mean())]

        slices = [f.squeeze() for f in vols]
        nes.plot.slices_grayscale(slices, rot90=0, titles=titles, grid=[1, len(vols)])


def epoch_sample_models(models, filenames, rng, x, mx,
                        x_orig=None, idx=None, y=None,
                        x_others=None,
                        load_per_model=None,
                        by_name=False, grid=None,
                        single_image=False,
                        do_plot=True,
                        clip=None,
                        do_mix=False):
    print('Warning: this modifies models weights', file=sys.stderr)

    if not isinstance(models, (list, tuple)):
        models = [models]

    if not isinstance(filenames, (list, tuple)):
        filenames = [filenames]

    data = get_data_example(x, mx, x_orig=x_orig, idx=idx, ret_idx=True)

    if rng is None:
        rng = [None]

    for ri, i in enumerate(rng):
        if i is not None:
            print('Epoch %d' % i)

        data = get_data_example(x, mx, x_orig=x_orig, idx=idx, ret_idx=True)
        sidx = data[-1]
        data = data[:-1]
        idx = sidx  # fix idx from here on

        vols, titles = _init_vols_titles(data, sidx, y, x_orig, x_others)

        # add model results
        for mi, model in enumerate(models):
            if (i is not None) or (load_per_model is not None):
                try:
                    if load_per_model is not None:
                        model.load_weights(filenames[mi].format(
                            epoch=load_per_model[mi]), by_name=by_name)
                        print(model.name, filenames[mi].format(epoch=load_per_model[mi]))
                    else:
                        model.load_weights(filenames[mi].format(epoch=i), by_name=by_name)

                except Exception as e:
                    print("Model %s outputs might be meaningless" % model.name, file=sys.stderr)
                    print(str(e), file=sys.stderr)

            res = model.predict(list(data[0:2]))
            if isinstance(res, (list, tuple)):
                res = res[0]
            if clip is not None:
                res = np.clip(res, clip[0], clip[1])

            vols += [res]
            titles += ['%s %3.4f %3.4f' % (model.name, _mse(res, data[2]),
                                           _mse(res * data[1], data[0] * data[1]) / data[1].mean())]

            if do_mix:
                vols += [data[0] * data[1] + res * (1 - data[1])]
                titles += ['mix']

        slices = [f.squeeze() for f in vols]
        if grid is None:
            grid = [1, len(vols)]

        if do_plot:
            nes.plot.slices_grayscale(slices, rot90=3, titles=titles, grid=grid)

        if single_image:
            slices = np.concatenate(slices, 1)

    print(titles)
    return slices


def epoch_models_loss(models, filenames, rng, x, mx, x_orig,
                      batch_size=32, nb_samples=None, by_name=False,
                      loss=None,
                      load_per_model=None,
                      do_plot=True,
                      clip=None):
    print('Warning: this modifies models weights', file=sys.stderr)
    if loss is None:
        loss = lambda x, y: ((x - y) ** 2).mean()

    if not isinstance(models, (list, tuple)):
        models = [models]

    if not isinstance(filenames, (list, tuple)):
        filenames = [filenames]

    if nb_samples is not None:
        data = get_data_example(x, mx, x_orig=x_orig, rand_seed=0, nb_samples=nb_samples)
    else:  # all data
        data = [x, mx, x_orig]
        nb_samples = x.shape[0]

    stats = np.zeros((len(rng), len(models), nb_samples))
    for ri, i in enumerate(rng):
        print('Epoch %d' % i)

        # load model results
        for mi, model in enumerate(models):
            try:
                if load_per_model:
                    model.load_weights(filenames[mi].format(
                        epoch=load_per_model[mi]), by_name=by_name)
                else:
                    model.load_weights(filenames[mi].format(epoch=i), by_name=by_name)

            except Exception as e:
                print("Model %s outputs might be meaningless" % model.name, file=sys.stderr)
                print(str(e), file=sys.stderr)

            res = []
            for bi in range(0, nb_samples, batch_size):
                j = np.minimum(nb_samples, bi + batch_size)
                data_i = [f[bi:j, :] for f in data]
                res_i = model.predict(data_i[:2])

                if isinstance(res_i, (list, tuple)):
                    res_i = res_i[0]
                res += [res_i]
            res = np.concatenate(res, 0)

            if clip is not None:
                res = np.clip(res, clip[0], clip[1])

            stats[ri, mi, :] = [loss(res[k, :], data[-1][k, :]) for k in range(nb_samples)]

    # do some plotting
    if do_plot:
        all_model_names = [f.name for f in models]
        means = np.mean(stats, -1)
        stdevs = np.std(stats, -1)
        ylim = [np.min(means[-1, :]) - np.mean(stdevs), np.max(means[-1, :]) + np.mean(stdevs)]
        # plt.plot(rng, means, '.-')
        plt.figure(figsize=(20, 5))
        for mi in range(len(all_model_names)):
            plt.errorbar(rng, means[:, mi], yerr=stdevs[:, mi])
        plt.legend(all_model_names)
        plt.ylim(ylim)
        plt.show()

        df = stats - stats[:, 0, np.newaxis, :]
        means = np.mean(df, -1)
        stdevs = np.std(df, -1)
        ylim = [np.min(means[-1, :]) - np.mean(stdevs), np.max(means[-1, :]) + np.mean(stdevs)]
        plt.figure(figsize=(20, 5))
        for mi in range(len(all_model_names)):
            plt.errorbar(rng, means[:, mi], yerr=stdevs[:, mi])
        plt.legend(all_model_names)
        plt.ylim(ylim)
        plt.show()

    return stats


def models(model, run, data, activation='relu', mid_dropout=None, ignore_models=[], conv_dropout=0):
    # seg_models
    models = nes.utils.seg.seg_models(model, run, data,
                                      enc_lambda_layers=[],
                                      seed=0,
                                      activation=activation,
                                      enc_batch_norm=None,
                                      batch_norm=None)

    models['sparse-conv-enc'], im_tensor, mask_tensor = \
        conv_enc(model.nb_features,
                 (*run.patch_size, 1),
                 model.nb_levels,
                 model.conv_size,
                 name='sparse-conv-enc',
                 feat_mult=model.feat_mult,
                 pool_size=model.pool_size,
                 activation=activation,
                 conv_dropout=conv_dropout,
                 use_residuals=model.use_residuals,
                 nb_conv_per_level=model.nb_convs_per_level,
                 nargout=3)

    # conv-only model
    if 'sparse-conv-ae' not in ignore_models:
        models['sparse-conv-ae'] = ne.utils.stack_models(
            [models['sparse-conv-enc'], models['img-img-%s-ae' % model.ae_type][0]], [[0]])
        models['sparse-conv-ae'].name = 'sparse-conv-ae'

    if 'sparse-conv-vae' not in ignore_models:

        models_conv_enc_2, mu, mask_tensor_2, logvar, mask_tensor_2_ = \
            conv_enc(model.nb_features,
                     (*run.patch_size, 1),
                     model.nb_levels,
                     model.conv_size,
                     double_last_conv_layer=True,
                     name='sparse-conv-enc',
                     feat_mult=model.feat_mult,
                     pool_size=model.pool_size,
                     activation=activation,
                     use_residuals=model.use_residuals,
                     conv_dropout=conv_dropout,
                     nb_conv_per_level=model.nb_convs_per_level,
                     nargout=5)
        # split model last layer into two and aff back sample
        # sh = models_conv_enc_2.outputs[0].get_shape().as_list()[-1]//2
        # mu = KL.Lambda(lambda x: x[...,:sh], name='conv-split-mu')(models_conv_enc_2.outputs[0])
        # logvar = KL.Lambda(lambda x: x[...,sh:],
        #  name='conv-split-logvar')(models_conv_enc_2.outputs[0])
        mu_flat = KL.Flatten(name='mu_flat')(mu)
        logvar_flat = KL.Flatten(name='logvar_flat')(logvar)
        sample = KL.Lambda(ne.models._VAESample().sample_z, name='sample_z')([mu_flat, logvar_flat])
        sample_reshape = KL.Reshape(
            models['img-img-%s-ae' % model.ae_type][0].inputs[0].get_shape().as_list()[1:])(sample)
        tmp_model = keras.models.Model(models_conv_enc_2.inputs, sample_reshape)

        models['sparse-conv-vae'] = ne.utils.stack_models(
            [tmp_model, models['img-img-dense-ae'][0]], [[0]])
        models['sparse-conv-vae'] = \
            keras.models.Model(models['sparse-conv-vae'].inputs,
                               [models['sparse-conv-vae'].outputs[0], mu_flat, logvar_flat])
        # need to add another layer block of the type in the conv
        models['sparse-conv-vae'].name = 'sparse-conv-vae'

    # conv-dense ae
    if 'sparse-dense-ae' not in ignore_models:
        mydense_layer = nes.layers.SpatiallySparse_Dense(
            models['sparse-conv-enc'].output.get_shape()[1:].as_list(), model.enc_size[0])
        dense = mydense_layer([im_tensor, mask_tensor])
        if mid_dropout is not None:
            dense = KL.Dropout(mid_dropout, name='dropout')(dense)
        updense = mydense_layer([dense])
        tmp_ae_dense_model = keras.models.Model(models['sparse-conv-enc'].inputs, updense)
        models['sparse-dense-ae'] = ne.utils.stack_models(
            [tmp_ae_dense_model, models['img-img-dense-ae'][0]], [[0]])
        models['sparse-dense-ae'].name = 'sparse-dense-ae'

    if 'sparse-dense-vae' not in ignore_models:
        mydense_layer = \
            nes.layers.SpatiallySparse_Dense(
                models['sparse-conv-enc'].output.get_shape()[1:].as_list(),
                model.enc_size[0], name='sparse-mu')
        # mydense_layer_logvar =
        #   nes.layers.SpatiallySparse_Dense(
        #       models['sparse-conv-enc'].output.get_shape()[1:].as_list(), model.enc_size[0],
        #       name='sparse-logvar', use_bias=True)
        mydense_layer_logvar = KL.Dense(model.enc_size[0], name='sparse-logvar')
        dense = mydense_layer([im_tensor, mask_tensor])
        # dense_logvar = mydense_layer_logvar([im_tensor, mask_tensor])
        dense_logvar = mydense_layer_logvar(KL.Flatten()(im_tensor))
        if mid_dropout is not None:
            dense = KL.Dropout(mid_dropout, name='dropout')(dense)
            dense_logvar = KL.Dropout(mid_dropout, name='dropout')(dense_logvar)
        sample = KL.Lambda(ne.models._VAESample().sample_z, name='sample_z')(
            [dense, dense_logvar])  # sample
        # sample = KL.Reshape(models['sparse-conv-enc'].output.get_shape()[1:].as_list())(sample)
        updense = mydense_layer([sample])
        tmp_ae_dense_model = keras.models.Model(models['sparse-conv-enc'].inputs, updense)
        models['sparse-dense-vae'] = ne.utils.stack_models(
            [tmp_ae_dense_model, models['img-img-dense-ae'][0]], [[0]])
        models['sparse-dense-vae'] = \
            keras.models.Model(models['sparse-dense-vae'].inputs,
                               [models['sparse-dense-vae'].outputs[0], dense, dense_logvar])
        models['sparse-dense-vae'].name = 'sparse-dense-vae'

    # linear model
    if 'sparse-linear-ae' not in ignore_models:
        im = KL.Input((*run.patch_size, 1))
        mask_input = KL.Input((*run.patch_size, 1))
        model_inputs = [im, mask_input]
        mydense_layer = nes.layers.SpatiallySparse_Dense(
            im.get_shape()[1:].as_list(), model.enc_size[0], use_bias=True)
        dense = mydense_layer([im, mask_input])
        final_im = mydense_layer([dense])
        models['sparse-linear-ae'] = keras.models.Model(model_inputs, final_im)
        models['sparse-linear-ae'].name = 'sparse-linear-ae'

    # interpolation model
    if 'conv-ae-zero' not in ignore_models:
        models['img-img-conv-ae-single'] = ne.utils.stack_models(
            [models['img-img-%s-ae' % model.ae_type][f] for f in [-1, 0]], [[0], [0]])
        models['conv-ae-zero'] = keras.models.clone_model(models['img-img-conv-ae-single'])
        # add the mask so that the loss can have access to it -- note that it's not used in the
        # convolutions.
        minp = KL.Input((*run.patch_size, 1), name='sparse-ae-mask')
        models['conv-ae-zero'] = keras.models.Model(
            [models['conv-ae-zero'].input, minp], models['conv-ae-zero'].output)
        models['conv-ae-zero'].name = 'conv-ae-zero'

    # interpolation model
    if 'conv-vae-zero' not in ignore_models:
        ae_type = model.ae_type
        model.ae_type = 'conv'
        enc_size = model.enc_size
        model.enc_size = [None] * (len(run.patch_size) + 1)  # should just propagate data
        models2 = nes.utils.seg.seg_models(model, run, data,
                                           enc_lambda_layers=[],
                                           seed=0,
                                           nb_input_features=1,
                                           activation=activation,
                                           enc_batch_norm=None,
                                           batch_norm=None)
        model.ae_type = ae_type
        model.enc_size = enc_size

        mu_layer = [f.name for f in models2['img-img-conv-vae-single'].layers
                    if f.name.endswith('mu_enc')][0]
        mu_out = KL.Flatten()(
            models2['img-img-conv-vae-single'].get_layer(mu_layer).get_output_at(-1))
        logvar_layer = [f.name for f in models2['img-img-conv-vae-single'].layers
                        if f.name.endswith('sigma_enc')][0]
        logvar_out = KL.Flatten()(
            models2['img-img-conv-vae-single'].get_layer(logvar_layer).get_output_at(-1))

        models['conv-vae-zero'] = models2['img-img-conv-vae-single']
        # add the mask so that the loss can have access to it --
        # note that it's not used in the convolutions.
        minp = KL.Input((*run.patch_size, 1), name='sparse-vae-mask')
        models['conv-vae-zero'] = \
            keras.models.Model([models['conv-vae-zero'].input, minp],
                               [models['conv-vae-zero'].output, mu_out, logvar_out])
        models['conv-vae-zero'].name = 'conv-vae-zero'

        for layer in models['conv-vae-zero'].layers:
            layer.name = layer.name.replace('img-img-conv', 'img-img-%s' % model.ae_type)

    # interpolation model
    if 'conv-ae-mean' not in ignore_models:
        models['img-img-conv-ae-single'] = ne.utils.stack_models(
            [models['img-img-%s-ae' % model.ae_type][f] for f in [-1, 0]], [[0], [0]])
        models['conv-ae-mean'] = keras.models.clone_model(models['img-img-conv-ae-single'])
        # add the mask so that the loss can have access to it -- note that it's not used in the
        # convolutions.
        minp = KL.Input((*run.patch_size, 1), name='sparse-ae-mask')
        models['conv-ae-mean'] = keras.models.Model(
            [models['conv-ae-mean'].input, minp], models['conv-ae-mean'].output)
        models['conv-ae-mean'].name = 'conv-ae-mean'

    # interpolation model
    if 'conv-vae-mean' not in ignore_models:
        ae_type = model.ae_type
        model.ae_type = 'conv'
        enc_size = model.enc_size
        model.enc_size = [None] * (len(run.patch_size) + 1)  # should just propagate data
        models2 = nes.utils.seg.seg_models(model, run, data,
                                           enc_lambda_layers=[],
                                           seed=0,
                                           nb_input_features=1,
                                           activation=activation,
                                           enc_batch_norm=None,
                                           batch_norm=None)
        model.ae_type = ae_type
        model.enc_size = enc_size

        mu_layer = [f.name for f in models2['img-img-conv-vae-single'].layers
                    if f.name.endswith('mu_enc')][0]
        mu_out = KL.Flatten()(
            models2['img-img-conv-vae-single'].get_layer(mu_layer).get_output_at(-1))
        logvar_layer = [f.name for f in models2['img-img-conv-vae-single'].layers
                        if f.name.endswith('sigma_enc')][0]
        logvar_out = KL.Flatten()(
            models2['img-img-conv-vae-single'].get_layer(logvar_layer).get_output_at(-1))

        models['conv-vae-mean'] = models2['img-img-conv-vae-single']
        # add the mask so that the loss can have access to it --
        #   note that it's not used in the convolutions.
        minp = KL.Input((*run.patch_size, 1), name='sparse-vae-mask')
        models['conv-vae-mean'] = \
            keras.models.Model([models['conv-vae-mean'].input, minp],
                               [models['conv-vae-mean'].output, mu_out, logvar_out])
        models['conv-vae-mean'].name = 'conv-vae-mean'

        for layer in models['conv-vae-mean'].layers:
            layer.name = layer.name.replace('img-img-conv', 'img-img-%s' % model.ae_type)

    # interpolation model
    if 'conv-ae-simple' not in ignore_models:
        models['img-img-conv-ae-single'] = ne.utils.stack_models(
            [models['img-img-%s-ae' % model.ae_type][f] for f in [-1, 0]], [[0], [0]])
        models['conv-ae-simple'] = keras.models.clone_model(models['img-img-conv-ae-single'])
        # add the mask so that the loss can have access to it -- note that it's not used in the
        # convolutions.
        minp = KL.Input((*run.patch_size, 1), name='sparse-ae-mask')
        models['conv-ae-simple'] = keras.models.Model(
            [models['conv-ae-simple'].input, minp], models['conv-ae-simple'].output)
        models['conv-ae-simple'].name = 'conv-ae-simple'

    # interpolation model
    if 'conv-vae-simple' not in ignore_models:
        ae_type = model.ae_type
        model.ae_type = 'conv'
        enc_size = model.enc_size
        model.enc_size = [None] * (len(run.patch_size) + 1)  # should just propagate data
        models2 = nes.utils.seg.seg_models(model, run, data,
                                           enc_lambda_layers=[],
                                           seed=0,
                                           nb_input_features=1,
                                           activation=activation,
                                           enc_batch_norm=None,
                                           batch_norm=None)
        model.ae_type = ae_type
        model.enc_size = enc_size

        mu_layer = [f.name for f in models2['img-img-conv-vae-single'].layers
                    if f.name.endswith('mu_enc')][0]
        mu_out = KL.Flatten()(
            models2['img-img-conv-vae-single'].get_layer(mu_layer).get_output_at(-1))
        logvar_layer = [f.name for f in models2['img-img-conv-vae-single'].layers
                        if f.name.endswith('sigma_enc')][0]
        logvar_out = KL.Flatten()(
            models2['img-img-conv-vae-single'].get_layer(logvar_layer).get_output_at(-1))

        models['conv-vae-simple'] = models2['img-img-conv-vae-single']
        # add the mask so that the loss can have access to it -- note that
        # it's not used in the convolutions.
        minp = KL.Input((*run.patch_size, 1), name='sparse-vae-mask')
        models['conv-vae-simple'] = \
            keras.models.Model([models['conv-vae-simple'].input, minp],
                               [models['conv-vae-simple'].output, mu_out, logvar_out])
        models['conv-vae-simple'].name = 'conv-vae-simple'

        for layer in models['conv-vae-simple'].layers:
            layer.name = layer.name.replace('img-img-conv', 'img-img-%s' % model.ae_type)

    # interpolation model
    if 'conv-ae-simple-wmask' not in ignore_models:
        models2 = nes.utils.seg.seg_models(model, run, data,
                                           enc_lambda_layers=[],
                                           seed=0,
                                           nb_input_features=2,
                                           activation=activation,
                                           enc_batch_norm=None,
                                           batch_norm=None)

        tmp_mod_1 = ne.utils.stack_models(
            [models2['img-img-%s-ae' % model.ae_type][f] for f in [-1, 0]], [[0], [0]])
        models['conv-ae-simple-wmask'] = keras.models.clone_model(tmp_mod_1)
        # add the mask so that the loss can have access to it -- note that it's not used in the
        # convolutions.
        im_input = KL.Input((*run.patch_size, 1), name='conv-ae-simple-wmask-wmask-im')
        minp = KL.Input((*run.patch_size, 1), name='conv-ae-simple-wmask-mask')
        conc = KL.Concatenate(name='concat')([im_input, minp])
        tmp_model = keras.models.Model([im_input, minp], conc)

        q = keras.models.Model(models['conv-ae-simple-wmask'].input,
                               models['conv-ae-simple-wmask'].output)
        models['conv-ae-simple-wmask'] = ne.utils.stack_models([tmp_model, q], [[0]])
        models['conv-ae-simple-wmask'].name = 'conv-ae-simple-wmask'

    # interpolation model
    if 'conv-vae-simple-wmask' not in ignore_models:
        ae_type = model.ae_type
        model.ae_type = 'conv'
        enc_size = model.enc_size
        model.enc_size = [None] * (len(run.patch_size) + 1)  # should just propagate data
        models2 = nes.utils.seg.seg_models(model, run, data,
                                           enc_lambda_layers=[],
                                           seed=0,
                                           nb_input_features=2,
                                           activation=activation,
                                           enc_batch_norm=None,
                                           batch_norm=None)
        model.ae_type = ae_type
        model.enc_size = enc_size

        mu_layer = [f.name for f in models2['img-img-conv-vae-single'].layers
                    if f.name.endswith('mu_enc')][0]
        mu_out = KL.Flatten()(
            models2['img-img-conv-vae-single'].get_layer(mu_layer).get_output_at(-1))
        logvar_layer = [f.name for f in models2['img-img-conv-vae-single'].layers
                        if f.name.endswith('sigma_enc')][0]
        logvar_out = KL.Flatten()(
            models2['img-img-conv-vae-single'].get_layer(logvar_layer).get_output_at(-1))

        models['conv-vae-simple-wmask'] = models2['img-img-conv-vae-single']

        im_input = KL.Input((*run.patch_size, 1), name='conv-vae-simple-wmask-wmask-im')
        minp = KL.Input((*run.patch_size, 1), name='conv-vae-simple-wmask-mask')
        conc = KL.Concatenate(name='concat')([im_input, minp])
        tmp_model = keras.models.Model([im_input, minp], conc)

        models['conv-vae-simple-wmask'] = \
            keras.models.Model(models['conv-vae-simple-wmask'].input,
                               [models['conv-vae-simple-wmask'].output, mu_out, logvar_out])
        models['conv-vae-simple-wmask'] = ne.utils.stack_models(
            [tmp_model, models['conv-vae-simple-wmask']], [[0]])
        models['conv-vae-simple-wmask'].name = 'conv-vae-simple-wmask'

        for layer in models['conv-vae-simple-wmask'].layers:
            layer.name = layer.name.replace('img-img-conv', 'img-img-%s' % model.ae_type)

    # interpolation model
    # note: moved 'sparse-ae' to 'sparse-conv-dense-ae'
    if 'sparse-conv-dense-ae' not in ignore_models:
        models['sparse-conv-dense-ae'] = keras.models.clone_model(models['img-img-dense-ae-single'])
        # add the mask so that the loss can have access to it --
        # note that it's not used in the convolutions.
        minp = KL.Input((*run.patch_size, 1), name='sparse-ae-mask')
        models['sparse-conv-dense-ae'] = keras.models.Model(
            [models['sparse-conv-dense-ae'].input, minp], models['sparse-conv-dense-ae'].output)
        models['sparse-conv-dense-ae'].name = 'sparse-conv-dense-ae'

    # interpolation model
    if 'sparse-conv-dense-vae' not in ignore_models:
        # cloning VAE models has issues :(
        model_clones = nes.utils.seg.seg_models(model, run, data,
                                                enc_lambda_layers=[],
                                                seed=0,
                                                activation=activation,
                                                enc_batch_norm=None,
                                                batch_norm=None)
        models['sparse-conv-dense-vae'] = model_clones['img-img-dense-vae-single']
        minp = KL.Input((*run.patch_size, 1), name='sparse-vae-mask')
        mu = models['sparse-conv-dense-vae'].get_layer(
            'img-img-dense-vae_ae_mu_enc_dense_%d' % model.enc_size[0]).get_output_at(1)
        sigma = models['sparse-conv-dense-vae'].get_layer(
            'img-img-dense-vae_ae_sigma_enc_dense_%d' % model.enc_size[0]).get_output_at(1)
        models['sparse-conv-dense-vae'] = \
            keras.models.Model([models['sparse-conv-dense-vae'].input, minp],
                               [models['sparse-conv-dense-vae'].outputs[0], mu, sigma])
        models['sparse-conv-dense-vae'].name = 'sparse-conv-dense-vae'

    # interpolation model with mask
    if 'sparse-ae-mask' not in ignore_models:
        models2 = nes.utils.seg.seg_models(model, run, data,
                                           enc_lambda_layers=[],
                                           seed=0,
                                           nb_input_features=2,
                                           activation=activation,
                                           enc_batch_norm=None,
                                           batch_norm=None)
        q = keras.models.clone_model(models2['img-img-%s-ae-single' % model.ae_type])
        im_input = KL.Input((*run.patch_size, 1), name='sparse-ae-wmask-im')
        mask_input = KL.Input((*run.patch_size, 1), name='sparse-ae-wmask-mask')
        conc = KL.Concatenate(name='concat')([im_input, mask_input])
        tmp_model = keras.models.Model([im_input, mask_input], conc)
        models['sparse-ae-wmask'] = ne.utils.stack_models([tmp_model, q], [[0]])
        models['sparse-ae-wmask'].name = 'sparse-ae-wmask'

    # interpolation model with mask
    if 'sparse-vae-mask' not in ignore_models:
        models2 = nes.utils.seg.seg_models(model, run, data,
                                           enc_lambda_layers=[],
                                           seed=0,
                                           nb_input_features=2,
                                           activation=activation,
                                           enc_batch_norm=None,
                                           batch_norm=None)
        q = models2['img-img-dense-vae-single']
        im_input = KL.Input((*run.patch_size, 1), name='sparse-vae-wmask-im')
        mask_input = KL.Input((*run.patch_size, 1), name='sparse-vae-wmask-mask')
        conc = KL.Concatenate(name='concat')([im_input, mask_input])
        tmp_model = keras.models.Model([im_input, mask_input], conc)
        models['sparse-vae-wmask'] = ne.utils.stack_models([tmp_model, q], [[0]])

        mu = models['sparse-vae-wmask'].get_layer('img-img-dense-vae_ae_mu_enc_dense_%d' %
                                                  model.enc_size[0]).get_output_at(2)
        sigma = models['sparse-vae-wmask'].get_layer(
            'img-img-dense-vae_ae_sigma_enc_dense_%d' % model.enc_size[0]).get_output_at(2)
        models['sparse-vae-wmask'] = \
            keras.models.Model(models['sparse-vae-wmask'].inputs,
                               [*models['sparse-vae-wmask'].outputs, mu, sigma])
        models['sparse-vae-wmask'].name = 'sparse-vae-wmask'

    return models


def subj_gen_3D_brain(path,
                      batch_size=1,
                      maskthr=0.5,
                      start=[64, 64, 64],
                      shape=(128, 128, 128),
                      randdel=(0, 0, 0),
                      max_files=np.inf,
                      rand_seed=0,
                      mode=None,
                      sparsity=0.86,
                      line_rot_max_angle=15,
                      yield_iso=False,
                      only_iso=False,
                      sge_subj=None,
                      normalize=False,
                      mask_zero_in=0,
                      neck_mask=None):
    griddata = scipy.interpolate.griddata

    def _crop_and_prep(vol, this_start, shape):
        vol = nd.volcrop(vol, start=this_start, end=[this_start[f] + shape[f] for f in range(3)])
        vol = vol[np.newaxis, ..., np.newaxis]
        return vol

    subjects = [f for f in os.listdir(path) if "_fake" not in f]
    if rand_seed is not None:
        random.Random(rand_seed).shuffle(subjects)

    # assert rand_seed is not None and rand_seed==0, "need this for consistent masks"

    idx = 0

    while 1:
        vols = []
        isovols = []
        masks = []
        for bidx in range(batch_size):

            subj = subjects[idx]
            if sge_subj is not None:
                print('SGE SPECIFIC INDEX', sge_subj)
                subj = sge_subj

            if mode == 'orig':
                vol = np.load(os.path.join(path, subj))['vol_data']

                if yield_iso:
                    isovol = np.copy(vol)

                mask_name = os.path.join(path, subj + '_fakemask_%d_%d.npz' %
                                         (sparsity * 100, line_rot_max_angle))
                ds_name = os.path.join(path, subj + '_fakeds_%d_%d.npz' %
                                       (sparsity * 100, line_rot_max_angle))
                if not os.path.isfile(mask_name):
                    pad_amt = 20
                    print('creating mask')

                    line_density = np.maximum(1, np.round(1 / (1 - sparsity))).astype(int)

                    vol = np.pad(vol, pad_amt, mode='edge')
                    volshape = vol.shape
                    ri = np.random.randint(0, line_density)
                    r = np.random.randint(-line_rot_max_angle, line_rot_max_angle, 2)
                    vol = scipy.ndimage.interpolation.rotate(
                        vol, -r[0], axes=(0, 1), reshape=False, order=1)
                    vol = scipy.ndimage.interpolation.rotate(
                        vol, -r[1], axes=(1, 2), reshape=False, order=1)

                    # ds volumes
                    vol_pre_sample = np.copy(vol)
                    vol = vol[:, ri::line_density, :]

                    # upsample

                    vol = scipy.ndimage.zoom(vol, [1, line_density, 1], order=1)
                    if line_density > 1:
                        vol = vol[:, :-(line_density - 1), :]
                    vol = np.pad(vol, [(0, 0), (ri, 0), (0, 0)], mode='constant')
                    vol = np.pad(vol, [(0, 0), (0, volshape[1] - vol.shape[1]),
                                       (0, 0)], mode='constant')
                    vol[:, ri::line_density, :] = vol_pre_sample[:, ri::line_density, :]

                    mask = vol * 0
                    mask[:, ri::line_density, :] = 1

                    # rotate back
                    vol = scipy.ndimage.interpolation.rotate(
                        vol, r[1], axes=(1, 2), reshape=False, order=1)
                    vol = scipy.ndimage.interpolation.rotate(
                        vol, r[0], axes=(0, 1), reshape=False, order=1)
                    mask = scipy.ndimage.interpolation.rotate(
                        mask, r[1], axes=(1, 2), reshape=False, order=1)
                    mask = scipy.ndimage.interpolation.rotate(
                        mask, r[0], axes=(0, 1), reshape=False, order=1)

                    # crop pad
                    vol = vol[pad_amt:-pad_amt, pad_amt:-pad_amt, pad_amt:-pad_amt]
                    mask = mask[pad_amt:-pad_amt, pad_amt:-pad_amt, pad_amt:-pad_amt]

                    np.savez_compressed(ds_name, vol_data=vol)
                    np.savez_compressed(mask_name, vol_data=mask)
                else:
                    vol = np.load(ds_name)['vol_data']
                    mask = np.load(mask_name)['vol_data']

            else:
                # vol = np.load(os.path.join(path, subj, '%s_ds5_us5_reg.npz' % subj))['vol_data']
                # mask = np.load(os.path.join(path, subj, '%s_ds5_us5_dsmask_reg.npz' % subj))
                #   ['vol_data']

                # mmap mode!
                if only_iso:
                    vol = np.load(os.path.join(
                        path, subj, '%s_iso_2_ds5_us5_size_reg.npy' % subj), mmap_mode='r')
                else:
                    vol = np.load(os.path.join(path, subj, '%s_ds5_us5_reg.npy' %
                                               subj), mmap_mode='r')
                    mask = np.load(os.path.join(
                        path, subj, '%s_ds5_us5_dsmask_reg.npy' % subj), mmap_mode='r')

            # print(os.path.join(path, subj, '%s_ds5_us5_reg.npy' % subj))

            # prep cropping range
            if start is None:
                this_start = [np.random.randint(0, vol.shape[f] - shape[f]) for f in range(3)]
            else:
                this_start = list(start)

            for d in range(len(this_start)):
                if randdel[d] > 0:
                    this_start[d] += np.random.randint(-randdel[d], randdel[d])

            vol = _crop_and_prep(vol, this_start, shape)
            if only_iso:
                mask = vol * 0 + 1
            else:
                mask = _crop_and_prep(mask, this_start, shape)

            vols = [*vols, vol]

            if yield_iso and not (mode == 'orig'):
                isovol = np.load(os.path.join(
                    path, subj, '%s_iso_2_ds5_us5_size_reg.npy' % subj), mmap_mode='r')

            if yield_iso:
                isovol = _crop_and_prep(isovol, this_start, shape)
                isovols = [*isovols, isovol]

            # some more processing for mask
            if maskthr is not None:
                mask = (mask > maskthr)
                mask = mask.astype(float)  # separate line for timing
            if neck_mask is not None:
                mask2 = np.ones(mask.shape)
                mask2[:, :, :, :neck_mask, :] = 0
                # mask2[:, 16:-16, 16:-16, 16:-16, :] = 1  # ALL MASK 1s!
                mask = mask2 * mask  # np.logical_and(mask , mask2).astype(float)

            masks = [*masks, mask]
            idx += 1
            if idx == len(subjects) or idx >= max_files:
                idx = 0  # TODO: change to MOD
                # random.Random(rand_seed).shuffle(subjects)

        if len(vols) > 1:
            vol = np.concatenate(vols, 0)
        else:
            vol = vols[0]
        if yield_iso:
            isovol = np.concatenate(isovols, 0)
        if len(vols) > 1:
            mask = np.concatenate(masks, 0)
        else:
            mask = masks[0]

        if mask_zero_in > 0:
            m = mask_zero_in
            crp = ((0, 0), (m, m), (m, m), (m, m), (0, 0))
            mask2 = np.zeros(mask.shape)
            mask2[:, m:-m, m:-m, m:-m, :] = mask[:, m:-m, m:-m, m:-m, :]
            mask = mask2

            # slower:
            # mask = nd.volcrop(mask, crop=crp)
            # mask = np.pad(mask, crp, 'constant')

        if np.any(np.sum(mask.reshape((mask.shape[0], -1)), 1) < 10):
            print('skipping this, got ~0 mask')
            continue

        assert vol.shape[0] == batch_size
        assert mask.shape[0] == batch_size

        if normalize:
            np.percentile
            vol -= vol.min((1, 2, 3, 4))
            vol /= np.percentile(vol, 99.9, axis=(1, 2, 3, 4))
            # vol -= vol.mean((1,2,3,4), keepdims=True)
            # vol /= vol.std((1,2,3,4), keepdims=True)
            if yield_iso:
                isovol -= isovol.min((1, 2, 3, 4))
                isovol /= np.percentile(isovol, 99.9, axis=(1, 2, 3, 4))
                # isovol -= isovol.mean((1,2,3,4), keepdims=True)
                # isovol /= isovol.std((1,2,3,4), keepdims=True)

        if yield_iso:
            yield ([vol, mask], vol, isovol)
        else:
            yield ([vol, mask], vol)


def group_bar(data, colors=None, width=0.35, title=None, dataset_names=None, model_names=None,
              figsize=(10, 5), ylabel='MSE', ylim=None, y_rotation=None, xlabel='', legend_loc=1):

    if not isinstance(data, (list, tuple)):
        data = [data[f, :, :] for f in range(data.shape[0])]

    # nb_dataset x nb_models x nb_datapoints
    nb_datasets = len(data)
    nb_models = data[0].shape[0]
    if colors is None:
        colors = [np.random.random(3) for _ in range(nb_models)]
    assert len(colors) >= nb_models, "Need at least color for each model %d %d" % (
        len(colors), nb_models)
    if dataset_names is not None:
        assert len(dataset_names) == nb_datasets, "not enough dataset names"
    if model_names is not None:
        assert len(model_names) == nb_models, \
            "number of model_names %d should be equal to nb models %d" % (
            len(model_names), nb_models)
    ind = np.arange(nb_datasets)    # the x locations for the groups
    # if width is None:
    # width = 0.35         # the width of the bars

    fig, ax = plt.subplots(figsize=figsize)

    ps = [None] * nb_models
    for mi in range(nb_models):
        means = [f[mi, :].mean() for f in data]
        stds = [f[mi, :].std() for f in data]
        ps[mi] = ax.bar(ind + width * mi, means, width, color=colors[mi], bottom=0, yerr=stds)

    ax.set_xticks(ind + (nb_models - 1) * width / 2)
    if dataset_names is not None:
        ax.set_xticklabels(dataset_names)

    if model_names is not None:
        ax.legend(ps, model_names, loc=legend_loc, bbox_to_anchor=(1.05, 1))
    # ax.yaxis.set_units(inch)
    ax.autoscale_view()

    if ylim is not None:
        plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if y_rotation is not None:
        plt.xticks(ind + (nb_models - 1) * width / 2, dataset_names, rotation='vertical')

    plt.show()
    return fig


def _init_vols_titles(data, sidx, y, x_orig, x_others=None):
    vols = []
    titles = []
    if x_orig is not None:
        vols = [data[2]]
        titles = ['im (%d-%d)' % (sidx, y[sidx])] if y is not None else ['im (%d)' % (sidx)]

    vols += [data[0], data[1]]
    titles += ['sparse',
               'mask % 3.2f' % data[1].mean()]

    if x_others is not None:
        if not isinstance(x_others, (list, tuple)):
            x_others = [x_others]
        x_others_vols = [f[sidx, np.newaxis, ...] for f in x_others]
        vols += x_others_vols
        titles += ['other (%3.2f %3.2f)' % (_mse(f, vols[0]),
                                            _mse(f * data[1], data[0] * data[1]) / data[1].mean())
                   for f in x_others_vols]

    return vols, titles


def _mse(x, y, axis=None, wt=None):
    if wt is None:
        wt = x * 0 + 1
    return (wt * ((x - y)**2)).mean(axis=axis)


def _mae(x, y, axis=None):
    return (np.abs(x - y)).mean(axis=axis)
