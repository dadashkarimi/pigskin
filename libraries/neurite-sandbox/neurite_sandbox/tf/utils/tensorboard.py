"""
Tensorboard utilities
"""

# built in
import os
import sys

# third party
import numpy as np
import scipy
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorboard.backend.event_processing import event_accumulator
import matplotlib
import matplotlib.pyplot as plt

# our libraries
import pystrum.pynd.ndutils as nd
import neurite as ne


def plot_loss_values(tb_path,
                     loss_names=None,
                     subplots=None,
                     figsize=None,
                     log_loss=False,
                     ylim=None,
                     xlim=None,
                     same_axis=True,
                     ylim_last_steps=None,
                     verbose=False):
    """
    plot losses from tensorboard files

    Parameters:
        tb_path: path to tensorboard event files. All files are assumed to start with 'events'
        loss_names: a list/tuple with loss names (strings)
        subplots: the subplot array. default: (1, len(loss_names))
        figsize: the figure size in in (figsize parameter to plt.figure()). default: (15, 4)
        log_loss: bool whether to use log on the y (loss) axis.

    Example:
        loss_names = ['loss', 'val_loss']
        sb_tb.plot_loss_values(tb_path, loss_names,
                               subplots=(1, 2), figsize=(20, 10), log_loss=True)
    """

    # get all available losses
    all_loss_names = get_loss_names(tb_path)
    if verbose:
        print('loss names:')
        for loss_name in all_loss_names:
            print('\t%s' % loss_name)

    # handle inputs
    if subplots is None:
        subplots = [1, len(loss_names)]
    if figsize is None:
        figsize = (15, 4)
    if loss_names is None:
        loss_names = all_loss_names

    # get the loss values of the loss names
    loss_vals = get_loss_values(tb_path, loss_names)

    plt.figure(figsize=figsize)

    if ylim_last_steps is not None:
        # assert xlim is None, "xlim cannot be specified when same_axis is True"
        assert ylim is None, "ylim cannot be specified when same_axis is True"
        ylim = [np.inf, -np.inf]

    if same_axis:
        assert xlim is None, "xlim cannot be specified when same_axis is True"
        assert ylim_last_steps or ylim is None, "ylim cannot be specified when same_axis is True"
        # assert ylim_last_steps is None,
        #   "ylim_last_steps cannot be specified when same_axis is True"
        ylim = [np.inf, -np.inf]
        xlim = [np.inf, -np.inf]

    # gather data
    data = {}
    for li, loss in enumerate(loss_names):

        x = loss_vals[loss + '_steps']
        y = loss_vals[loss]

        xi = np.argsort(x)
        x = np.sort(x)
        y = [y[f] for f in xi]

        if log_loss:
            y = np.log(y)

        data[loss] = [x, y]

        if same_axis and len(y) > 0:
            if ylim_last_steps is None:
                ylim = [np.minimum(ylim[0], np.min(y)), np.maximum(ylim[1], np.max(y))]
            xlim = [np.minimum(xlim[0], np.min(x)), np.maximum(xlim[1], np.max(x))]

        if ylim_last_steps is not None and len(y) > 0:
            ymn = y[-np.minimum(ylim_last_steps, len(y)):]
            ylim = [np.minimum(ylim[0], np.min(ymn)), np.maximum(ylim[1], np.max(ymn))]

        if len(y) == 0:
            ylim = [0, 0]
            xlim = [0, 0]

    for li, loss in enumerate(loss_names):
        x, y = data[loss]

        ylabel = '(loss)'
        if log_loss:
            ylabel = 'log (loss)'

        # plot
        ax = plt.subplot(*subplots, li + 1)
        plt.plot(x, y)
        ax.grid(True)

        plt.title(loss)
        plt.xlabel('steps')
        plt.ylabel(ylabel)

        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)

    plt.show()


def get_loss_values(tb_path, loss_names):
    """
    Get the loss data for several losses in the tensorboard path
    """

    tb_files = _get_tb_files(tb_path)

    losses = {}
    for loss in loss_names:
        losses[loss] = []
        losses[loss + '_steps'] = []

    for file in tb_files:
        ea = event_accumulator.EventAccumulator(os.path.join(tb_path, file))
        ea.Reload()
        for loss in loss_names:
            if loss in ea.Tags()['scalars']:
                # print(ea.Scalars(loss))
                sc = [f.value for f in ea.Scalars(loss)]
                losses[loss] = [*losses[loss], *sc]
                st = [f.step for f in ea.Scalars(loss)]
                losses[loss + '_steps'] = [*losses[loss + '_steps'], *st]

        # debug code to look at
        # if 'loss' in ea.Tags()['scalars'] and len(ea.Scalars('loss')) > 0:
            # print(file, ea.Scalars('loss')[-1].step)

    return losses


def get_loss_names(tb_path):
    """
    get a list of the loss names saved in tensorboard files at a particular location
    """

    tb_files = _get_tb_files(tb_path)

    keys = []
    for file in tb_files:
        ea = event_accumulator.EventAccumulator(os.path.join(tb_path, file))
        ea.Reload()
        keys = [*keys, *ea.Tags()['scalars']]

    return list(set(keys))


def _get_tb_files(tb_path, startswith='events'):
    return [f for f in os.listdir(tb_path) if f.startswith(startswith)]
