#!/usr/bin/env python3
import os
import glob
import argparse

import numpy as np
import nibabel as nib
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

import voxelmorph as vxm
import surfa as sf
from scipy import ndimage as ndi

import param_3d  # your existing config module with img_size_192, etc.


print("Available devices:", tf.config.list_physical_devices())


# ------------------------------------------------
# Model loading (fixed absolute path)
# ------------------------------------------------
def get_pig_model():
    """
    Loads the 3D U-Net pig model trained with GMM labels.
    Expects weights in: /cbica/home/dadashkj/decipher/models_gmm_6_6/weights_*.h5
    """
    epsilon = 1e-7
    min_max_norm = Lambda(
        lambda x: (x - K.min(x)) / (K.max(x) - K.min(x) + epsilon) * 1.0
    )

    print("Loading pig model (GMM)...")
    en = [16, 16, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    de = [64, 64, 64, 64, 64, 64, 64, 64, 64, 16, 16, 2]

    input_img = Input(
        shape=(
            param_3d.img_size_192,
            param_3d.img_size_192,
            param_3d.img_size_192,
            1,
        )
    )

    unet_model = vxm.networks.Unet(
        inshape=(
            param_3d.img_size_192,
            param_3d.img_size_192,
            param_3d.img_size_192,
            1,
        ),
        nb_features=(en, de),
        nb_conv_per_level=2,
        final_activation_function="softmax",
    )

    # ---- fixed absolute folder for weights ----
    weight_dir = "/cbica/home/dadashkj/decipher/models_gmm_6_6"
    weight_pattern = os.path.join(weight_dir, "weights_*.h5")

    latest_weight = max(
        glob.glob(weight_pattern),
        key=os.path.getctime,
        default=None,
    )

    if latest_weight is None:
        raise FileNotFoundError(
            f"No weights found matching pattern: {weight_pattern}"
        )

    print("Using weight file:", latest_weight)

    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)

    return combined_model


# ------------------------------------------------
# Simple refinement: largest component + hole filling
# ------------------------------------------------
def refine_prediction2(crop_img, model):
    """
    Given a [192,192,192,1] input image and the model,
    returns a binary mask (uint8) of shape [192,192,192].

    This mirrors the logic from your notebook:
    - argmax over classes
    - keep largest connected component
    - fill holes
    """
    # [1, 192,192,192,1] -> [1,192,192,192,C] probs
    prediction_one_hot = model.predict(crop_img[None, ...], verbose=0)
    # take argmax over channels -> [192,192,192]
    initial_prediction = np.argmax(prediction_one_hot, axis=-1)[0]

    # Largest connected component of foreground
    labeled, num_components = ndi.label(initial_prediction > 0)
    if num_components == 0:
        return np.zeros_like(initial_prediction, dtype=np.uint8)

    sizes = ndi.sum(initial_prediction > 0, labeled, index=range(num_components + 1))
    # skip background index 0
    largest_label = np.argmax(sizes[1:]) + 1
    largest_mask = labeled == largest_label

    # Fill holes
    largest_mask_filled = ndi.binary_fill_holes(largest_mask)

    final_pred = (largest_mask_filled > 0).astype(np.uint8)
    return final_pred


# ------------------------------------------------
# Main CLI
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PIGSKIN inference: segment pig brain mask from a T1 NIfTI."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input image NIfTI (.nii or .nii.gz)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output mask NIfTI (.nii.gz recommended)",
    )
    parser.add_argument(
        "--resize_to_1",
        action="store_true",
        help="Resample voxels to 1mm isotropic using surfa (like in your notebook).",
    )

    args = parser.parse_args()

    # Load model (weights are at fixed absolute path)
    pig_model = get_pig_model()

    # ------------------------------------------------
    # Load input image
    # ------------------------------------------------
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    print("Loading image:", args.input)
    image = sf.load_volume(args.input)

    if args.resize_to_1:
        print("Resizing image to 1mm isotropic voxels...")
        image = image.resize(1)
    
    # Save original geometry to map prediction back
    orig_shape = image.shape
    affine = np.array(image.geom.vox2world)
    
    print("Original image shape:", orig_shape)
    
    # 🔹 NEW: resample to 192×192×192 for the network (like in your notebook)
    image_192 = image.reshape(
        [
            param_3d.img_size_192,
            param_3d.img_size_192,
            param_3d.img_size_192,
        ]
    )
    
    print("Resampled image shape for network:", image_192.shape)
    
    # Shape -> [192,192,192,1] for the model
    crop_img = image_192.reshape(
        [
            param_3d.img_size_192,
            param_3d.img_size_192,
            param_3d.img_size_192,
            1,
        ]
    )


    # ------------------------------------------------
    # Run inference
    # ------------------------------------------------
    print("Running model inference...")
    prediction = refine_prediction2(crop_img, pig_model)

    # Convert to Volume then back to numpy (if you ever want to use surfa later)
    prediction_vol = sf.Volume(prediction.astype(np.int32)).reshape(orig_shape)
    prediction_np = prediction_vol.data

    # ------------------------------------------------
    # Save output mask
    # ------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir != "" and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    nib.save(
        nib.Nifti1Image(prediction_np.astype(np.int32), affine),
        args.output,
    )

    print("Saved mask to:", args.output)


if __name__ == "__main__":
    main()
