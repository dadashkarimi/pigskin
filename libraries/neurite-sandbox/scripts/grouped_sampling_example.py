#!/usr/bin/env python3

# Example showing how to synthesize images from label maps, linking certain
# labels such that their intensities lie within a range from one another.


import numpy as np
import nibabel as nib
import tensorflow as tf
import neurite_sandbox as nes


# Read all training label maps.
label_map = '/autofs/cluster/fssubjects/test/mu40-crazy/data/affine/lab.nii.gz'
label_map = np.int32(nib.load(label_map).dataobj)
label_map = np.expand_dims(label_map, axis=(0, -1))


# Extract individual labels.
labels = np.unique(label_map)
num_label = len(labels)


# Sampling ranges for label means. The first is BG.
mean_min = (0, *[25] * (num_label - 1))
mean_max = (0, *[225] * (num_label - 1))


# Define rules: label of interest, label to link to, maximim relative distance.
linkages = (
    (2, 41, 0.01),
    (0, 41, 0.00),
)


# Wrap function.
def example_rules(shape, dtype, seed):
    '''Wrap nes.utils.generative.sample_linked to link label intensities.'''
    return nes.utils.generative.sample_linked(
        shape, labels, linkages, mean_min, mean_max, dtype, seed,
    )


# Create model.
model = nes.models.labels_to_image(
    labels_in=labels,
    in_shape=label_map.shape[1:-1],
    mean_func=example_rules,
    zero_background=0,
    return_mean=True
)


# Generate example.
*_, mean_image = model.predict(label_map)

out = np.squeeze(mean_image)
out = nib.MGHImage(out, affine=None)
nib.save(out, filename='out.mgz')
