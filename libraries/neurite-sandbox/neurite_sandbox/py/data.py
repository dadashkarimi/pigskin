"""
utilities for dealing with data in neurite sandbox
"""


# python imports
import os
import random

# third party
import matplotlib.pylab as plt
from tqdm import tqdm
import numpy as np
import nibabel as nib

# specific imports that used to be here
from neurite.py.data import split_dataset


def load_image_data(path,
                    nb_max=None,
                    tqdm=tqdm,
                    ext=None,
                    randomize=False,
                    rand_seed=None,
                    stack=True):
    """
    load and return images (large numpy array or list)
    data supported is anything supported by load_image()

    Parameters:
        path: the path to the folder from which to read data
        nb_max: max number of elements to load
        tqdm: decorator to display progress. in jupyters, recommend tqdm_notebook
        ext: extensions of the files if you want to only load certain data
        randomize: whether to randommize data
        rand_seed: random seed for randomization
        stack: whether to stack and return a numpy array, or return a list
    """

    # get files
    files = [f for f in os.listdir(path)]
    if ext is not None:
        files = [f for f in files if f.endswith(ext)]

    # randomize
    if randomize:
        if rand_seed:
            random.seed(rand_seed)
        random.shuffle(files)

    # prep tqdm decorator
    if tqdm is not None:
        files = tqdm(files)

    # load images
    ims = []
    for file in files:
        im = load_image(os.path.join(path, file))
        ims.append(im)

        if nb_max is not None and nb_max <= len(ims):
            break

    # stack into one array
    if stack:
        ims = np.stack(ims, 0)

    return ims


def load_image(filename, npz_varname=None):
    """
    load and return single image

    Available formats:
        image (jpg/jpeg/png)
        npy
        npz (provide npz_varname)
    """

    # get extension
    base, ext = os.path.splitext(filename)

    if ext == '.gz':
        ext = os.path.splitext(base)[-1]
        full_ext = ext + '.gz'

    if ext in ['.jpg', '.png', '.jpeg']:
        im = plt.imread(filename)

        if len(im.shape) > 2 and im.shape[2] > 1:
            assert im.shape[2] == 3, 'expecting RGB image'
            im = _rgb2gray(im.astype(float))

    elif ext in ['.npz']:
        loaded_file = np.load(filename)
        if npz_varname is None:
            assert len(loaded_file.keys()) == 1, \
                "need to provide npz_varname for file {} since several found".format(filename)
            npz_varname = list(loaded_file.keys())[0]
        im = loaded_file[npz_varname]

    elif ext in ['.npy']:
        im = np.load(filename)

    elif ext in ['.nii']:
        im = nib.load(filename).get_data().astype('float')

    else:
        raise Exception('extension %s not understood' % ext)

    return im


def _rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
