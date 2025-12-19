import os
import glob
import csv
import numpy as np
import nibabel as nib
import surfa as sf
from tqdm import tqdm
from tensorflow.keras.layers import Lambda
from keras import backend as K
import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt
import scipy.ndimage as ndi
import tensorflow.keras.layers as KL
import voxelmorph as vxm

import argparse
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pathlib
import re
import json
from keras import backend as K
import param_3d
import data
import model_3d
from data_3d import *
import scipy.ndimage as ndimage

import nibabel as nib
from tqdm import tqdm
from tensorflow.keras.layers import Lambda
from utils import *
from help import *
import os

from tensorflow.keras.models import Model


# ==== Argument Parsing ====
parser = argparse.ArgumentParser(description="Evaluate weights for a given number of trainings.")
parser.add_argument('--num_trainings', type=int, default=18,
                    help='Number of trainings (e.g., epoch count) used for model folder and CSV naming.')
args = parser.parse_args()

# ==== Flags ====
orig = False
num_trainings = args.num_trainings
t1 = False
t2 = False
unique = False
num_files = 1
injury = True

test_on_injury = None
resize_to_1=True

# ==== Parameters ====
k1, k2 = 6, 6
# CHANGE THIS PATH TO THE NEW DATASET LOCATION
validation_folder_path = "/cbica/home/dadashkj/synthstrip_data_v1.5"

# --- The logic for picking subfolders is now based on the new path ---
if test_on_injury is True:
    # Pick folders with 'month', 'day', or 'post' in the name
    subfolders = sorted([
        f.name for f in os.scandir(validation_folder_path)
        if f.is_dir() and any(keyword in f.name.lower() for keyword in ['month', 'day', 'post'])
    ])
elif test_on_injury is False:
    # Pick folders ending with 'pre'
    subfolders = sorted([
        f.name for f in os.scandir(validation_folder_path)
        if f.is_dir() and f.name.lower().endswith('pre')
    ])
else:
    # None → use all subject folders
    subfolders = sorted([f.name for f in os.scandir(validation_folder_path) if f.is_dir()])


def build_output_csv(t1, t2, num_trainings, train_suffix, test_suffix, resize_to_1=False):
    """
    Build output CSV filename.
    """
    return "dice_scores_human.csv"

# This function is now simplified and always returns the same string.
output_csv = build_output_csv(t1, t2, num_trainings, None, None, resize_to_1=resize_to_1)
print("Output CSV:", output_csv)

# ==== Dynamic model folder logic ====
def get_model_folder():
    if resize_to_1:
        return "models_gmm_6_6"
    if orig:
        return f"models_gmm_{k1}_{k2}_orig"
    elif num_trainings:
        prefix = f"models_gmm_new_{num_trainings}"
        if unique:
            prefix += "_unique"
        elif t1:
            prefix += "_t1"
            if injury:
                prefix += "_injury"
        elif t2:
            prefix += "_t2"
            if injury:
                prefix += "_injury"
        return f"{prefix}_{k1}_{k2}"
    elif t1:
        return f"models_gmm_t1_{k1}_{k2}"
    elif t2:
        return f"models_gmm_t2_{k1}_{k2}"
    else:
        return f"models_gmm_{k1}_{k2}"

model_folder = get_model_folder()
print("#######################################################", model_folder)

# --- DEBUG PRINTS ---
print(f"\nDebug: Calculated model_folder: {model_folder}")
if not os.path.exists(model_folder):
    print(f"Debug: ERROR: Model folder DOES NOT EXIST: {model_folder}")
    print("Debug: Please ensure your current working directory is correct, or provide an absolute path.")
else:
    print(f"Debug: Model folder EXISTS: {model_folder}")

debug_raw_glob_paths = glob.glob(os.path.join(model_folder, "weights_*.h5"))
print(f"Debug: Raw glob found {len(debug_raw_glob_paths)} files matching 'weights_*.h5' in {model_folder}")
if not debug_raw_glob_paths:
    print("Debug: ERROR: No .h5 weight files found by glob. Check path and filename pattern (e.g., 'weights_*.h5').")
    print(f"Debug: Expected pattern: {os.path.join(model_folder, 'weights_*.h5')}")

# ---- sort by epoch number, NOT ctime ----
# ---- sort by file modification time ----
# --- collect all weights in the correct folder ---
weight_paths = glob.glob(os.path.join(model_folder, "weights_*.h5"))

if not weight_paths:
    print(f"⚠️ No weights found in {model_folder}")
    latest_weight = None
else:
    # sort alphabetically (timestamps sort correctly this way)
    weight_paths = sorted(weight_paths)
    print(f"Found {len(weight_paths)} weights in {model_folder}")
    print(f"Oldest: {os.path.basename(weight_paths[0])}")
    print(f"Newest: {os.path.basename(weight_paths[-1])}")

    if num_files == 1:
        # take newest only
        latest_weight = weight_paths[-1]
        weight_paths = [latest_weight]
        print(f"✅ Using latest weight: {os.path.basename(latest_weight)}")
    else:
        # evenly sample multiple weights if requested
        indices = np.linspace(0, len(weight_paths) - 1, num_files)
        indices = np.unique(np.round(indices).astype(int))
        weight_paths = [weight_paths[i] for i in indices]
        print(f"✅ Uniformly selected {len(weight_paths)} weights: {[os.path.basename(w) for w in weight_paths]}")

print(f"Debug: After epoch-number sorting (oldest to newest), weight_paths has {len(weight_paths)} entries.")
if weight_paths:
    print(f"Debug: Oldest weight found: {os.path.basename(weight_paths[0])}")
    print(f"Debug: Newest weight found: {os.path.basename(weight_paths[-1])}")
else:
    print("Debug: No weight paths found.")

# --- LOGIC TO UNIFORMLY SELECT NUM_FILES ---
if len(weight_paths) > num_files:
    indices = np.linspace(0, len(weight_paths) - 1, num_files)
    indices = np.unique(np.round(indices).astype(int))
    selected_weight_paths = [weight_paths[i] for i in indices]
    weight_paths = selected_weight_paths

    print(f"Debug: After uniformly selecting {num_files} files, weight_paths has {len(weight_paths)} entries.")
    if weight_paths:
        print(f"Debug: First uniformly selected weight path: {os.path.basename(weight_paths[0])}")
        print(f"Debug: Last uniformly selected weight path: {os.path.basename(weight_paths[-1])}")
    else:
        print("Debug: No weight paths remaining after uniform selection (should not happen).")
elif len(weight_paths) > 0 and len(weight_paths) <= num_files:
    print(f"Debug: Total weight files ({len(weight_paths)}) <= num_files ({num_files}). Processing all available files.")
else:
    print("Debug: No weight paths found initially by glob. The evaluation loop will not run.")

# ==== Model loader ====
def load_model_for_weight(weight_path):
    epsilon = 1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x) + epsilon) * 1.0)
    en = [16, 16, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    de = [64, 64, 64, 64, 64, 64, 64, 64, 64, 16, 16, 2]
    input_img = tf.keras.Input(shape=(param_3d.img_size_192,) * 3 + (1,))
    unet_model = vxm.networks.Unet(inshape=input_img.shape[1:], nb_features=(en, de),
                                   nb_conv_per_level=2, final_activation_function='softmax')
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(weight_path)
    return combined_model

def refine_prediction2(crop_img, image, mask, mask2, model, folder,
                       orig_voxsize, suffix="",
                       new_image_size=(192, 192, 192), margin=0, cube_size=128):
    """
    Returns the final binary prediction mask (no file saving).
    """
    prediction_one_hot = model.predict(crop_img[None, ...], verbose=0)
    initial_prediction = np.argmax(prediction_one_hot, axis=-1)[0]

    labeled, num_components = ndi.label(initial_prediction > 0)
    if num_components == 0:
        return np.zeros_like(initial_prediction)

    largest_mask = labeled == np.argmax(ndi.sum(initial_prediction > 0, labeled, range(num_components + 1)))
    initial_prediction = ndi.binary_fill_holes(largest_mask)
    initial_prediction = (initial_prediction > 0).astype(np.int32)

    final_pred = (initial_prediction > 0).astype(np.uint8)
    return final_pred

# ==== Evaluation loop ====
results = []
test_suffixes = [""]
if weight_paths:
    # --- Load the latest model weight regardless of num_files ---
    # Assuming num_files is always 1, this simplifies the initial setup.
    latest_weight_path = weight_paths[-1]
    print(f"✅ Evaluating latest weight: {os.path.basename(latest_weight_path)}")
    model = load_model_for_weight(latest_weight_path)
    pig_model = model

    # --- Start a single evaluation loop for all subjects ---
    dice_scores_for_this_weight = []
    
    # The `test_suffixes` list from your original script is no longer necessary
    # since you are only checking for 'image.nii.gz'.
    
    for folder in subfolders:
        folder_path = os.path.join(validation_folder_path, folder)
        image_filename = os.path.join(folder_path, 'image.nii.gz')
        mask_filename = os.path.join(folder_path, 'mask.nii.gz')

        if not (os.path.isfile(image_filename) and os.path.isfile(mask_filename)):
            print(f"❌ Missing files in '{folder_path}' – skipping subject.")
            continue

        try:
            # Load original volumes
            image_orig = sf.load_volume(image_filename)
            mask_orig = sf.load_volume(mask_filename)

            # Apply resizing based on the flag
            if resize_to_1:
                image = image_orig.resize(1.0)
                mask = mask_orig.resize(1.0, method="nearest")
            else:
                image = image_orig
                mask = mask_orig
            
            # Reshape the image and mask after resizing
            crop_img = image.reshape([192, 192, 192, 1])
            mask = mask.reshape([192, 192, 192, 1])
            orig_voxsize = image_orig.geom.voxsize

        except Exception as e:
            print(f"Error loading volumes for '{folder_path}': {e} – skipping subject.")
            continue

        mask.data[mask.data != 0] = 1

        if np.sum(mask.data) < 1000:
            print(f"⚠️ Mask in '{folder_path}' too small ({int(np.sum(mask.data))} voxels) – skipping subject.")
            continue

        # Pass the pre-loaded model `pig_model`
        prediction = refine_prediction2(crop_img, image, mask, mask, pig_model, folder, orig_voxsize)

        mask_flat = mask.data.flatten()
        prediction_flat = prediction.flatten() > 0

        denominator = (np.sum(mask_flat) + np.sum(prediction_flat))
        if denominator == 0:
            dice = 0.0
            print(f"Skipping Dice calculation for '{folder}': Denominator is zero. Assigning 0.0 Dice.")
        else:
            dice = 2 * np.sum(mask_flat * prediction_flat) / denominator

        dice_scores_for_this_weight.append(dice)
        print(f"Dice for {folder}: {dice:.4f}")

    # --- Final calculation and CSV logging after the inner loop finishes ---
    if dice_scores_for_this_weight:
        mean_dice = np.mean(dice_scores_for_this_weight)
        std_dice = np.std(dice_scores_for_this_weight)
        weight_basename = os.path.basename(latest_weight_path)
        print(f"✅ Average Dice for {weight_basename}: {mean_dice:.4f} ± {std_dice:.4f}")
        results.append((weight_basename, mean_dice, std_dice))
    else:
        print(f"⚠️ No valid dice scores were calculated for {os.path.basename(latest_weight_path)}.")

# ==== Final Save to CSV ====
if results:
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Weight_File_Name", "Average_Dice", "Std_Dice"])
        writer.writerows(results)
    print(f"\n✅ Finished. All results saved to: {output_csv}")
else:
    print(f"\n⚠️ No results were generated at all. The CSV file '{output_csv}' was not created or only contains a header.")
    print("Please check previous console output for 'Debug: ERROR', '❌ Missing file', or '⚠️ Mask too small' messages to understand why.")