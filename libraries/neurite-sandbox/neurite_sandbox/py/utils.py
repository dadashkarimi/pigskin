"""
python utilities for neurite sandbox
"""

# internal python imports
import os
import concurrent.futures

# third party imports
import numpy as np
import scipy

# local (our) imports


def get_backend():
    """
    Returns the currently used backend. Default is tensorflow unless the
    NEURITE_BACKEND environment variable is set to 'pytorch'.
    """
    return 'pytorch' if os.environ.get('NEURITE_BACKEND') == 'pytorch' else 'tensorflow'


def mat_print(A):
    """
    print a matrix in human readable format

    copied from StackOverflow solution
    """

    if A.ndim == 1:
        print(A)
    else:
        w = max([len(str(s)) for s in A])
        print(u'\u250c' + u'\u2500' * w + u'\u2510')
        for AA in A:
            print(' ', end='')
            print('[', end='')
            for i, AAA in enumerate(AA[:-1]):
                w1 = max([len(str(s)) for s in A[:, i]])
                print(str(AAA) + ' ' * (w1 - len(str(AAA)) + 1), end='')
            w1 = max([len(str(s)) for s in A[:, -1]])
            print(str(AA[-1]) + ' ' * (w1 - len(str(AA[-1]))), end='')
            print(']')
        print(u'\u2514' + u'\u2500' * w + u'\u2518')


def flatten_lists(lst):
    """                                                                         
    return a list of elements that are not lists                                

    if lst is not a list, will return [lst]                                     
    otherwise, will flatten any hierarchy to a long list of lists               
    """
    if not isinstance(lst, (tuple, list)):
        lst = [lst]

    long_lst = []
    for f in lst:
        if isinstance(f, (tuple, list)):
            long_lst.extend(flatten_lists(f))
        else:
            long_lst.append(f)
    return long_lst

def _nparray_to_list(npa):
    if npa is None:
        return None

    lst = []
    for bno in range(npa.shape[0]):
        lst.append(npa[bno,...])
    return lst


def _list_to_nparray(lst):
    if lst is None:
        return None

    npa = np.zeros((len(lst),) +  lst[0].shape)
    for bno in range(npa.shape[0]):
        npa[bno,...] = lst[bno]
    return npa

import random
def random_rotation_3d(image, max_angle, aux_ims=None):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    max_angle: `float`. The maximum rotation angle.

    Returns:
    batch of rotated 3D images
    """
    assert aux_ims == None, 'aux im rotation not implemented yet'

    size = image.shape
    image = np.squeeze(image)

    # rotate along z-axis
    angle = random.uniform(-max_angle, max_angle)
    image2 = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', axes=(0, 1), reshape=False)

    # rotate along y-axis
    angle = random.uniform(-max_angle, max_angle)
    image2 = np.transpose(image2, (0, 2, 1))
    image3 = scipy.ndimage.interpolation.rotate(image2, angle, mode='nearest', axes=(0, 1), reshape=False)
    image3 = np.transpose(image3, (0, 2, 1))  # undo first transpose
    
    # rotate along x-axis
    image3 = np.transpose(image3, (1, 2, 0))
    angle = random.uniform(-max_angle, max_angle)
    im_rot = scipy.ndimage.interpolation.rotate(image3, angle, mode='nearest', axes=(0, 1), reshape=False)
    im_rot = np.transpose(im_rot, (2, 0, 1))

    return im_rot.reshape(size)

def augment_image(ims, types=[], black_border=False, aux_ims=None, noise_max=.1, 
                  trans_fov_div=8, angle_max=15, aux_int=1, channels=0, spherical=False):
    """
    apply augmentations to given image
    if given list of images, apply the *same* augmentation to each image!
    aux_int - order of interpolation to use for aux images (use 0 for luts/segs)
    """

    in_shape = np.array(ims).shape

    if types is None or len(types) == 0:
        return ims    # nothing to do

    squeezed = False
    squeezed2 = False
    if type(ims) == np.ndarray and len(ims.shape) == 5:  # convert batch to list
        if ims.shape[0] == 1:
            squeezed2 = True

        if ims.shape[-1] == 1:
            squeezed = True
            ims = ims.squeeze()
            if aux_ims is not None:
                aux_ims = aux_ims.squeeze()

        ims_saved = ims
        ims = _nparray_to_list(ims[np.newaxis, ...])
        if aux_ims is not None:
            aux_ims = _nparray_to_list(aux_ims[np.newaxis,...])
        was_array = True
    else:
        was_array = False

    islist = isinstance(ims, (list, tuple))
    if not islist:
        if ims.shape[-1] == 1:
            squeezed = True
            ims = ims.squeeze()
        ims = [ims]

    if aux_ims is not None and not islist:
        if aux_ims.shape[-1] == 1 and squeezed:
            squeezed = True
            aux_ims = aux_ims.squeeze()
        aux_ims = [aux_ims]

    ndims = len(ims[0].shape) - channels

    if 'flip' in types and np.random.random() < 0.5:
        ims = [1 - f for f in ims]

    if 'flipaxis' in types and np.random.random() < 0.5:
        ims = [np.flip(f, 1) for f in ims]

    if 'contrast' in types:
        r = np.clip(np.random.normal(loc=1, scale=0.3), 0.3, 2)
        ims = [f**r for f in ims]

    if 'randconv' in types:
        k = np.random.randn(3, 3)
        ims = [scipy.signal.convolve2d(f, k, mode='same') for f in ims]

    if 'noise' in types:
        mag = np.random.uniform(0, noise_max)
        noise_ims = [mag * np.random.randn(*f.shape) for f in ims]
        ims = [f + noise_ims[fno] for fno, f in enumerate(ims)]
        # don't put noise into the aux images, just spatial transforms
        # if aux_ims is not None:
        #    aux_ims = [f+noise_ims[fno] for fno, f in enumerate(aux_ims)]

    if 'dc' in types:
        mag = np.random.uniform(0, noise_max)
        ims = [f + mag * np.random.randn(1) for f in ims]

    if 'randconv-one' in types:
        if ndims == 2:
            k = np.random.randn(3, 3) * 0.1  # small noise kernel
            k[1, 1] = 1  # with one in the center
            ims = [scipy.signal.convolve2d(f, k, mode='same') for f in ims]
        else:
            k = np.random.randn(3, 3, 3) * 0.1  # small noise kernel
            k[1, 1, 1] = 1  # with one in the center
            ims = [scipy.signal.convolve(f, k, mode='same') for f in ims]

    # Warning only apply rotation through this method if using for unsupervised. 
    # Otherwise also need to transform segs with same params.
    if 'rotate' in types:
        if ndims == 2:
            angle = np.random.uniform(-angle_max, angle_max)
            ims = [scipy.ndimage.rotate(f, angle, reshape=False, order=1) for f in ims]
            if aux_ims is not None and 'rotate':
                aux_ims = [scipy.ndimage.rotate(f, angle, reshape=False, order=aux_int) 
                           for f in aux_ims]
        else:
            ims = [random_rotation_3d(f, angle_max) for f in ims]


    # Warning only apply translation through this method if using for unsupervised. 
    # Otherwise also need to transform segs with same params.
    if 'translate' in types:
        sh = ims[0].shape[0:len(ims[0].shape) - channels]
        shift = [np.random.uniform(-f // trans_fov_div, f // trans_fov_div) for f in sh]
        if channels > 0:
            shift += [0] * channels

        if spherical:  # use spherical boundary conditions
            shift1 = [shift[0], 0, 0]
            ims1 = [scipy.ndimage.shift(f, shift1, order=1, mode='reflect') for f in ims]
            shift2 = [0, shift[1], 0]
            ims = [scipy.ndimage.shift(f, shift2, order=1, mode='wrap') for f in ims1]
            if aux_ims is not None:
                tmp = [scipy.ndimage.shift(f, shift1, order=1, mode='reflect') for f in aux_ims]
                aux_ims = [scipy.ndimage.shift(f, shift2, order=1, mode='wrap') for f in tmp]
        else:
            ims = [scipy.ndimage.shift(f, shift, order=0) for f in ims]
            if aux_ims is not None:
                aux_ims = [scipy.ndimage.shift(f, shift, order=0) for f in aux_ims]

    if black_border:
        for im in ims:
            if ndims == 3:
                im[:, 0] = 0
                im[0, :] = 0
                im[:, -1] = 0
                im[-1, :] = 0
            else:
                im[0, :, :] = 0
                im[-1, :, :] = 0
                im[:, 0, :] = 0
                im[:, -1, :] = 0
                im[:, :, 0] = 0
                im[:, :, -1] = 0

    if was_array:  # convert back to numpy array
        ims = _list_to_nparray(ims)
        aux_ims = _list_to_nparray(aux_ims)
        if squeezed:
            ims = ims[..., np.newaxis]
            if aux_ims is not None:
                aux_ims = aux_ims[..., np.newaxis]
        if squeezed2:
            ims = ims[np.newaxis, ...]
            if aux_ims is not None:
                aux_ims = aux_ims[np.newaxis, ...]

        ims = ims.reshape(in_shape)
        if aux_ims is not None:
            aux_ims = aux_ims.reshape(in_shape)

    if islist:
        if aux_ims is not None:
            return ims, aux_ims
        else:
            return ims
    else:
        if squeezed:
            ims[0] = ims[0][..., np.newaxis]
            if aux_ims is not None:
                aux_ims[0] = aux_ims[0][..., np.newaxis]

        if aux_ims is not None:
            return ims[0], aux_ims[0]
        else:
            return ims[0]


def mssd(x, y, labels, reduce=(np.mean, np.max), pool=None):
    """
    Compute symmetric surface distances between label maps.

    Arguments:
        x, y: Input label maps to compare, as NumPy arrays.
        labels: List of labels to consider.
        reduce: List of functions used for reduction. For example, choose
            np.mean for mean symmetric distances, np.max for Hausdorff.
        pool: Optional pool of workers for parallel computation. The function
            will use up to `len(labels)` workers in parallel. Has to be an
            instance of `concurrent.futures.Executor`. You should find a
            thread-backed pool faster than a process-backed pool.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879

    """
    # Inputs.
    x = np.squeeze(x)
    y = np.squeeze(y)
    if callable(reduce):
        reduce = [reduce]

    if pool:
        assert isinstance(pool, concurrent.futures.Executor), \
            'pool is not an instance of concurrent.futures.Executor'
        out = [pool.submit(_mssd_single, x, y, lab, reduce) for lab in labels]
        out = map(concurrent.futures.Future.result, out)

    else:
        out = map(lambda i: _mssd_single(x, y, i, reduce), labels)

    return tuple(map(np.asarray, zip(*out)))


def _mssd_single(x, y, label, reduce):
    """
    Surface-distance subroutine for internal consumption. See `mssd`.
    """
    # Binary masks.
    bw_1 = x == label
    bw_2 = y == label
    if np.count_nonzero(bw_1) == 0 or np.count_nonzero(bw_2) == 0:
        return [np.nan for _ in reduce]

    # Cropping.
    ind = np.nonzero(np.logical_or(bw_1, bw_2))  # Tuple: num_dim, num_nonzero.
    low = np.min(ind, axis=-1)
    upp = np.max(ind, axis=-1)
    ind = tuple(slice(a, u + 1) for a, u in zip(low, upp))
    bw_1 = bw_1[ind]
    bw_2 = bw_2[ind]

    # Inner contour and distance transform.
    cont_x = bw_1 & ~scipy.ndimage.binary_erosion(bw_1)
    cont_y = bw_2 & ~scipy.ndimage.binary_erosion(bw_2)
    dist_x = scipy.ndimage.distance_transform_edt(~cont_x)
    dist_y = scipy.ndimage.distance_transform_edt(~cont_y)

    # Distances.
    dist = np.concatenate((dist_x[cont_y], dist_y[cont_x]))
    return [red(dist) for red in reduce]


def erase_labels(img, seg, erase_list, erase_seg=False):
    '''
       erase a set of labels specified by the list erase_list. For every voxel in seg that is in erase_list
       set the corresponding locatino in img to 0. If erase_seg is True also set that location in seg
       to 0
    '''
    
    if len(erase_list) == 0:
        return img
    max_label = np.max(seg)
    map_labels = np.asarray([False if i in erase_list else True for i in range(max_label+1)])
    mask = map_labels[seg]
    img = img * mask  # make a copy of img, which *= does not do
    if erase_seg:
        seg = seg * mask    # make a copy of seg, which *= does not do
        return img, seg
    else:
        return img

