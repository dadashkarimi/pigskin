"""
neurite sandbox plotting tools
"""

# py imports
import sys
from imp import reload
import types

# third party
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# project-specific imports
import neurite as ne
import numpy as np
import neurite_sandbox as nes

import pystrum.pytools.plot as pytools_plot
import pystrum.pynd.ndutils as nd


def slices(slices_in,
           squeeze=True,
           rot90=0,
           add_labels=None,
           do_argmax=False,
           mid_slice_idx=None,
           lut=None,
           **kwargs):
    """
    like neurite.plot.slices, with some additional hacks
    """

    if do_argmax:  # needs to be before squeeze
        slices_in = [np.argmax(f, -1) for f in slices_in]

    if squeeze:
        slices_in = [np.squeeze(f) for f in slices_in]

    if len(slices_in[0].shape) == 3 and slices_in[0].shape[-1] > 3:
        if mid_slice_idx is None:
            mid_slice_idx = slices_in[0].shape[2] // 2
            print('assuming 3D. Taking middle slice %d out' % mid_slice_idx)
        else:
            print('assuming 3D. Using slice %d' % mid_slice_idx)
        slices_in = [f[:, :, mid_slice_idx] for f in slices_in]

    # rotate
    slices_in = [np.rot90(f, rot90) for f in slices_in]

    # add to slices_in
    if add_labels is not None:
        slices_in = nes.utils.seg.add_labels_to_slices(slices_in, add_labels)

    # plot
    fig, ax = ne.plot.slices(slices_in, **kwargs)
    return fig, ax


def slices_grayscale(slices_in, add_labels=[0, 1], squeeze=True, rot90=-1, **kwargs):
    if 'do_colorbars' not in kwargs:
        kwargs['do_colorbars'] = True
    if 'grid' not in kwargs:
        kwargs['grid'] = True
    if 'cmaps' not in kwargs:
        kwargs['cmaps'] = ['gray']

    slices_in = [f.astype(float) for f in slices_in]

    fig, ax = slices(slices_in, squeeze=squeeze, rot90=rot90,
                     add_labels=add_labels, **kwargs)
    return fig, ax


def slices_labels(slices_in, add_labels,
                  squeeze=True,
                  rot90=-1,
                  label_prob=None,
                  do_argmax=True,
                  **kwargs):
    """
    label_prob: a list of labels for which you want to see probability maps
    """
    # get colormap
    [ccmap, scrambled_cmap] = pytools_plot.jitter(len(add_labels), nargout=2)
    scrambled_cmap[0, :] = np.array([0, 0, 0, 1])
    ccmap = matplotlib.colors.ListedColormap(scrambled_cmap)

    if 'do_colorbars' not in kwargs:
        kwargs['do_colorbars'] = True
    if 'grid' not in kwargs:
        kwargs['grid'] = True
    prov_cmaps = True
    if 'cmaps' not in kwargs:
        prov_cmaps = False
        kwargs['cmaps'] = [ccmap]
    slices(slices_in, squeeze=squeeze, rot90=rot90, do_argmax=do_argmax,
           add_labels=add_labels, **kwargs)

    if label_prob is not None:
        for lp in label_prob:
            if not prov_cmaps:
                kwargs['cmaps'] = ['gray']
            slices_prob = [np.take(f, lp, -1).astype(float) for f in slices_in]
            slices(slices_prob, squeeze=squeeze, rot90=rot90,
                   add_labels=[0, 1], **kwargs)


def epoch_samples_ae_grayscale(model,
                               model_path,
                               gen,
                               nb_epochs,
                               nb_samples,
                               sample_type='linear',
                               single_sample='last',
                               first_epoch=0,
                               load_by_name=False,
                               **kwargs):
    """
    sample and visualize predictions from an autoencoder across epochs

    model: model to test
    model_path: path to load model from, should contain a '%d' to replace epochs
    gen: generator
    """

    print('please use `epoch_samples_ae`', file=sys.stderr)
    epoch_samples_ae(model,
                     model_path,
                     gen,
                     nb_epochs,
                     nb_samples,
                     sample_type=sample_type,
                     single_sample=single_sample,
                     first_epoch=first_epoch,
                     load_by_name=load_by_name,
                     **kwargs)


def epoch_samples_ae(model,
                     model_path,
                     gen,
                     nb_epochs,
                     nb_samples,
                     is_label_mask=False,
                     label_prob=None,
                     extract_single_sample=True,
                     add_labels=None,  # necessary if is_label_mask
                     sample_type='linear',
                     single_sample='last',
                     first_epoch=0,
                     load_by_name=False,
                     **kwargs):
    """
    sample and visualize predictions from an autoencoder across epochs

    model: model to test
    model_path: path to load model from, should contain a '%d' to replace epochs
    gen: generator
    """

    sample, rng, preds = nes.utils.seg.sample_across_epochs(model,
                                                            model_path,
                                                            gen,
                                                            nb_epochs,
                                                            nb_samples,
                                                            sample_type=sample_type,
                                                            single_sample=single_sample,
                                                            first_epoch=first_epoch,
                                                            load_by_name=load_by_name)

    print(model.name)
    pred_samples(sample, preds, rng,
                 is_label_mask=is_label_mask,
                 label_prob=label_prob,
                 extract_single_sample=extract_single_sample,
                 add_labels=add_labels,
                 **kwargs)

    return sample, rng, preds


def epoch_samples_vae(model,
                      model_path,
                      gen,
                      nb_epochs,
                      nb_samples,
                      is_label_mask=False,
                      label_prob=None,
                      extract_single_sample=True,
                      add_labels=None,  # necessary if is_label_mask
                      sample_type='linear',
                      single_sample='last',
                      first_epoch=0,
                      load_by_name=False,
                      figsize=(15, 3.5)):
    """
    sample and visualize predictions from an autoencoder across epochs

    model: model to test
    model_path: path to load model from, should contain a '%d' to replace epochs
    gen: generator
    """

    sample, rng, preds = epoch_samples_ae(model,
                                          model_path,
                                          gen,
                                          nb_epochs,
                                          nb_samples,
                                          is_label_mask=is_label_mask,
                                          label_prob=label_prob,
                                          extract_single_sample=extract_single_sample,
                                          add_labels=add_labels,
                                          sample_type=sample_type,
                                          single_sample=single_sample,
                                          first_epoch=first_epoch,
                                          load_by_name=load_by_name)

    # also plot mu and logvar
    print(len(sample[1]))
    plt.figure(figsize=figsize)
    if len(sample[1]) > 2:
        plt.subplot(1, 2, 1)
    plt.plot(rng, [np.mean(f[1]) for f in preds])
    plt.plot(rng, [np.max(f[1]) for f in preds])
    plt.title('mean and max mu')
    plt.xlabel('epochs')
    if len(sample[1]) > 2:
        plt.subplot(1, 2, 2)
        plt.plot(rng, [np.mean(f[2]) for f in preds])
        plt.plot(rng, [np.max(f[2]) for f in preds])
        plt.title('mean and max logvar')
        plt.xlabel('epochs')
    plt.show()

    return sample, preds


def sample_model(model,
                 gen,
                 is_label_mask=False,
                 label_prob=None,
                 extract_single_sample=True,
                 add_labels=None  # necessary if is_label_mask
                 ):
    """
    sample and visualize predictions from an autoencoder

    model: model to test
    gen: generator
    """

    if isinstance(gen, types.GeneratorType):
        sample = next(gen)
    else:
        sample = gen
    preds = [model.predict(sample[0])]
    rng = [1]

    # plot
    print(model.name)
    pred_samples(sample, preds, rng,
                 is_label_mask=is_label_mask,
                 label_prob=label_prob,
                 extract_single_sample=extract_single_sample,
                 add_labels=add_labels)


def pred_samples(sample, preds, rng,
                 is_label_mask=False,
                 label_prob=None,
                 extract_single_sample=True,
                 add_labels=None,  # necessary if is_label_mask
                 **kwargs
                 ):
    """
    visualize sample alongside predictions for that sample
    """

    # decide on true output based on
    if isinstance(sample[1], (list, tuple)):
        assert isinstance(preds[0], (list, tuple)), "sample and pred are inconsistent structures"
        sample_out = sample[1][0]
        preds = [f[0] for f in preds]
    else:
        assert not isinstance(preds[0], (list, tuple)), \
            "predictions are lists. Is the generator good?"
        sample_out = sample[1]

    if isinstance(sample[0], (list, tuple)):
        vols = [sample[0][0], sample_out, *preds]
    else:
        vols = [sample[0], sample_out, *preds]

    titles = ['true input', 'true output'] + \
        ['recon epoch %d' % f for f in rng]

    if preds[0].shape[0] > 1 and extract_single_sample:
        vols = [f[0, :] for f in vols]

    if not is_label_mask:
        slices_grayscale(vols, titles=titles, **kwargs)
    else:
        assert add_labels is not None, "add_labels must be non-None"
        slices_labels(vols, add_labels, titles=titles, label_prob=label_prob, **kwargs)


def pca(pca, x, y):
    x_mean = np.mean(x, 0)
    x_std = np.std(x, 0)

    W = pca.components_
    x_mu = W @ pca.mean_  # pca.mean_ is y_mean
    y_hat = x @ W + pca.mean_

    y_err = y_hat - y
    y_rel_err = y_err / np.maximum(0.5 * (np.abs(y) + np.abs(y_hat)), np.finfo('float').eps)

    plt.figure(figsize=(15, 7))
    plt.subplot(2, 3, 1)
    plt.plot(pca.explained_variance_ratio_)
    plt.title('var %% explained')
    plt.subplot(2, 3, 2)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.ylim([0, 1.01])
    plt.grid()
    plt.title('cumvar explained')
    plt.subplot(2, 3, 3)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.ylim([0.8, 1.01])
    plt.grid()
    plt.title('cumvar explained')

    plt.subplot(2, 3, 4)
    plt.plot(x_mean)
    plt.plot(x_mean + x_std, 'k')
    plt.plot(x_mean - x_std, 'k')
    plt.title('x mean across dims (sorted)')
    plt.subplot(2, 3, 5)
    plt.hist(y_rel_err.flat, 100)
    plt.title('y rel err histogram')
    plt.subplot(2, 3, 6)
    plt.imshow(W @ np.transpose(W), cmap=plt.get_cmap('gray'))
    plt.colorbar()
    plt.title('W * W\'')
    plt.show()


def pts_on_slices(slices_in,
                  surface_pts,  #
                  show=True,
                  bw_for_crop=None,
                  crop_pad=20,
                  bounding_box=None,
                  linespec=None,
                  **args):
    """
    surface_pts is a list of N x 2 elements
    if you want different surface_Pts for diff slices, it *has* to be a list of lists
    """

    # draw slices
    fig, axs = ne.plot.slices(slices_in, show=False, **args)

    # deal with data
    if linespec is None:
        linespec = ['.r', '.k', '.c']

    # deal with images
    if bw_for_crop is not None:
        assert bw_for_crop.ndim == 2, 'bw image is not 2D'
        assert bounding_box is None
        bounding_box = nd.boundingbox(bw_for_crop)
        bounding_box[0] = np.maximum(0, bounding_box[0] - crop_pad)
        bounding_box[1] = np.maximum(0, bounding_box[1] - crop_pad)
        bounding_box[2] = np.minimum(slices_in[0].shape[0], bounding_box[2] + crop_pad)
        bounding_box[3] = np.minimum(slices_in[0].shape[1], bounding_box[3] + crop_pad)
    else:
        bounding_box = [0, 0, *slices_in[0].shape]

    ylim = [bounding_box[0], bounding_box[2]]
    xlim = [bounding_box[1], bounding_box[3]]

    if not isinstance(surface_pts, (list, tuple)):
        surface_pts = [[surface_pts] for _ in range(len(axs))]

    # if surface_pts is a list but not a list of lists
    if not isinstance(surface_pts[0], (list, tuple)):
        surface_pts = [surface_pts] * len(slices_in)

    axs = axs.flatten()
    for i in range(len(axs)):
        for j in range(len(surface_pts[0])):
            axs[i].plot(surface_pts[i][j][:, 0], surface_pts[i][j][:, 1], linespec[j])
        axs[i].set_ylim(ylim)
        axs[i].set_xlim(xlim)
        # axs[i].axis('on')

    if show:
        plt.tight_layout()
        plt.show()

    return fig, axs


def image_overlay(img, lay, factors=[0.25, 0.75]):
    """
    overlay a grayscale image with a intensity map overlay
    """

    assert img.ndim == 2
    assert lay.ndim == 2

    # whiten images
    img -= img.min()
    img /= img.max()
    lay -= lay.min()
    lay /= lay.max()

    # grayscale img
    img = np.repeat(img[..., np.newaxis], 3, axis=img.ndim)

    # yellow img
    lay = np.repeat(lay[..., np.newaxis], 3, axis=lay.ndim)
    lay[..., 2] = 0

    return np.clip(img * factors[0] + lay * factors[1], 0, 1)


def flow(slices_in,  # the 2D slices
         titles=None,  # list of titles
         cmaps=None,  # list of colormaps
         width=15,  # width in in
         indexing='ij',  # whether to plot vecs w/ matrix indexing 'ij' or cartesian indexing 'xy'
         img_indexing=True,  # whether to match the image view, i.e. flip order of y axis
         grid=False,  # option to plot the images in a grid or a single row
         show=True,  # option to actually show the plot (plt.show())
         quiver_width=None,
         scale=1):  # note quiver essentially draws quiver length = 1/scale
    '''
    plot a grid of flows (2d+2 images)
    '''

    # input processing
    nb_plots = len(slices_in)
    for slice_in in slices_in:
        assert len(slice_in.shape) == 3, 'each slice has to be 3d: 2d+2 channels'
        assert slice_in.shape[-1] == 2, 'each slice has to be 3d: 2d+2 channels'

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    slices_in = np.copy(slices_in)  # Since img_indexing, indexing may modify slices_in in memory

    assert indexing in ['ij', 'xy']
    if indexing == 'ij':
        for si, slc in enumerate(slices_in):
            # Make y values negative to y-axis will point down in plot
            slices_in[si][:, :, 1] = -slices_in[si][:, :, 1]

    if img_indexing:
        for si, slc in enumerate(slices_in):
            slices_in[si] = np.flipud(slc)  # Flip vertical order of y values

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    scale = input_check(scale, nb_plots, 'scale')

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        u, v = slices_in[i][..., 0], slices_in[i][..., 1]
        colors = np.arctan2(u, v)
        colors[np.isnan(colors)] = 0
        norm = Normalize()
        norm.autoscale(colors)
        if cmaps[i] is None:
            colormap = cm.winter
        else:
            raise Exception("custom cmaps not currently implemented for plt.flow()")

        # show figure
        ax.quiver(u, v,
                  color=colormap(norm(colors).flatten()),
                  angles='xy',
                  units='xy',
                  width=quiver_width,
                  scale=scale[i])
        ax.axis('equal')

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)
    plt.tight_layout()

    if show:
        plt.show()

    return (fig, axs)
