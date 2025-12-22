
# local python imports

# third party
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
from tqdm import tqdm

# local
import neurite as ne
import voxelmorph as vxm
from pystrum import pynd


def warp_perlin_vol(vol_shape,
                    randwarp_shape=None,
                    randwarp_amt=10,
                    return_orig=False,
                    **kwargs):
    """
    tf: create a perlin noise volume and warp it.
    """

    # get the volume
    vol = ne.utils.perlin_vol(vol_shape, **kwargs)
    volo = vol

    # compute random warp
    if randwarp_shape is None:
        # TODO: NEED TO FIX PARAM HARDCODE of [10]
        randwarp_shape = [10] * len(vol_shape) + [len(vol_shape)]

    # create full-dimensional warp
    warp = K.random_normal(randwarp_shape, mean=0, stddev=randwarp_amt)
    warp = ne.utils.zoom(warp, [vol_shape[d] / randwarp_shape[d] for d in range(len(vol_shape))])

    # transform
    vol = vxm.utils.transform(vol, warp, interp_method='linear')[..., 0]
    vol = vol / tfp.stats.percentile(K.flatten(vol), 99)

    if return_orig:
        v2 = volo / tfp.stats.percentile(K.flatten(volo), 99)
        return vol, v2
    else:
        return vol


def prep_center_weighted_patch(patch_shape, min_wt=0.2):
    """
    prepare a volume that is min_wt in the center and 1 at the edges, 
    diffusing spherically
    """

    # start wt image
    patch_wt = np.zeros(patch_shape)
    center = [[f // 2] for f in patch_shape]
    patch_wt[center] = 1
    patch_wt = pynd.ndutils.bwdist(patch_wt)

    return np.maximum(patch_wt / patch_wt.max(), min_wt)


def get_perlin_volumes(vol_shape, fn, nb_labels,
                       warp_std_range=[0, 10],
                       do_background=True,
                       return_orig=False):
    """
    compute a series of volumes 
    """

    v = np.zeros(vol_shape + [nb_labels + int(do_background)])

    if return_orig:
        vo = np.zeros(vol_shape + [nb_labels + int(do_background)])

    if do_background:
        warp_std = np.random.uniform(*warp_std_range)
        if return_orig:
            v[..., 0], vo[..., 0] = fn([warp_std])[0]
        else:
            v[..., 0] = fn([warp_std])[0]

        q0 = prep_center_weighted_patch(vol_shape, min_wt=0)
        v[..., 0] = np.maximum(q0**2 * 2, v[..., 0] * q0 * 1.2)

    # go through labels
    for li in tqdm(range(int(do_background), nb_labels + int(do_background)), leave=False):
        warp_std = np.random.uniform(*warp_std_range)
        if return_orig:
            v[..., li], vo[..., li] = fn([warp_std])[0]
        else:
            v[..., li] = fn([warp_std])[0]

    if return_orig:
        return v, vo

    else:
        return v
