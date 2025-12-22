

# third party
import tensorflow as tf

# ours
import neurite_sandbox as nes
import voxelmorph as vxm


@tf.function
def blob_stack(vol_shape, radius, blur_sigma, amplitude, nb_blobs):

    sphere = nes.utils.sphere_vol(vol_shape, radius, dtype=tf.float32)[..., tf.newaxis]
    blobs = []
    for _ in range(nb_blobs):
        rand_field = nes.utils.random_smooth_deformation_field(vol_shape,
                                                               amplitude,
                                                               smooth_method='blur',
                                                               sigmas=blur_sigma)
        blobs.append(vxm.utils.transform(sphere, rand_field)[..., 0])

    return tf.stack(blobs, -1)


@tf.function
def blob_mix(vol_shape, radius, blur_sigma, amplitude, nb_blobs_per_image, min_wt=0.2, max_wt=0.2):
    blobs = blob_stack(vol_shape, radius, blur_sigma, amplitude, nb_blobs_per_image)
    wts = tf.random.uniform([1] * len(vol_shape) + [nb_blobs_per_image],
                            minval=min_wt,
                            maxval=max_wt)
    wts = wts / tf.reduce_sum(wts)
    sum_blobs = tf.reduce_sum(blobs * wts, -1)
    return sum_blobs, blobs
