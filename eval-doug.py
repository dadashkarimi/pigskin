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

import argparse # <--- ADD THIS IMPORT
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


# ==== Argument Parsing ==== <--- ADD THIS SECTION
parser = argparse.ArgumentParser(description="Evaluate weights for a given number of trainings.")
parser.add_argument('--num_trainings', type=int, default=18, # Default to 18 if not provided, matches your common case
                    help='Number of trainings (e.g., epoch count) used for model folder and CSV naming.')
# You can add other flags here if you want them to be configurable from the command line:
# parser.add_argument('--t1', action='store_true', help='Set T1 flag (default is False).')
# parser.add_argument('--t2', action='store_true', default=True, help='Set T2 flag (default is True).')
# parser.add_argument('--injury', action='store_true', default=True, help='Set injury flag (default is True).')
# parser.add_argument('--num_files', type=int, default=10, help='Number of files to uniformly select (default is 10).')
# parser.add_argument('--k1', type=int, default=6, help='Parameter k1 (default is 6).')
# parser.add_argument('--k2', type=int, default=6, help='Parameter k2 (default is 6).')

args = parser.parse_args()

# ==== Flags ====
orig = False
num_trainings = args.num_trainings # <--- GET VALUE FROM COMMAND LINE ARGUMENT
t1 = False
t2 = True
unique = True
num_files = 1 # This remains hardcoded unless you add it to argparse
injury = True # This remains hardcoded unless you add it to argparse
test_on_injury=True
# ==== Parameters ====
k1, k2 = 6, 6 # These remain hardcoded unless you add them to argparse
validation_folder_path = "results_doug"
# subfolders = sorted([f.name for f in os.scandir(validation_folder_path) if f.is_dir()])

if test_on_injury:
    subfolders = sorted([f.name for f in os.scandir(validation_folder_path) if f.is_dir() and f.name.endswith('_post')])
else:
    subfolders = sorted([f.name for f in os.scandir(validation_folder_path) if f.is_dir() and f.name.endswith('_pre')])


# Define the single output CSV file name
output_csv = f"dice_scores.csv"
modality = "t1" if t1 else "t2" if t2 else ""
injury_suffix = "injury" if test_on_injury else ""
output_csv = f"dice_scores_doug_{modality}{injury_suffix}_{num_trainings}.csv"

# if t1:
#     output_csv = f"dice_scores_doug_t1_{num_trainings}.csv"
#     if injury:
#         output_csv = f"dice_scores_doug_t1_injury_{num_trainings}.csv"
# if t2:
#     output_csv = f"dice_scores_doug_t2_{num_trainings}.csv"
#     if injury:
#         output_csv = f"dice_scores_doug_t2_injury_{num_trainings}.csv"


# ==== Dynamic model folder logic ====
def get_model_folder():
    if orig:
        return f"models_gmm_{k1}_{k2}_orig"
    
    if num_trainings == 0:
        return f"models_gmm_{k1}_{k2}"

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



model_folder = get_model_folder()
latest_weight = max(glob.glob(os.path.join(model_folder, 'weights_epoch_*.h5')), key=os.path.getctime, default=None)

print("#######################################################",model_folder)
# --- DEBUG PRINTS (KEEP THESE!) ---
print(f"\nDebug: Calculated model_folder: {model_folder}")
if not os.path.exists(model_folder):
    print(f"Debug: ERROR: Model folder DOES NOT EXIST: {model_folder}")
    print("Debug: Please ensure your current working directory is the parent of this folder, or provide an absolute path.")
else:
    print(f"Debug: Model folder EXISTS: {model_folder}")

debug_raw_glob_paths = glob.glob(os.path.join(model_folder, "weights_epoch_*.h5"))
print(f"Debug: Raw glob found {len(debug_raw_glob_paths)} files matching 'weights_epoch_*.h5' in {model_folder}")
if not debug_raw_glob_paths:
    print("Debug: ERROR: No .h5 weight files found by glob. Check path and filename pattern (e.g., 'weights_epoch_*.h5').")
    print(f"Debug: Expected pattern: {os.path.join(model_folder, 'weights_epoch_*.h5')}")


weight_paths = sorted(debug_raw_glob_paths, key=os.path.getctime)

print(f"Debug: After initial sorting (oldest to newest), weight_paths has {len(weight_paths)} entries.")
if weight_paths:
    print(f"Debug: Oldest weight found: {os.path.basename(weight_paths[0])} (created: {os.path.getctime(weight_paths[0])})")
    print(f"Debug: Newest weight found: {os.path.basename(weight_paths[-1])} (created: {os.path.getctime(weight_paths[-1])})")


# --- LOGIC TO UNIFORMLY SELECT NUM_FILES ---
if len(weight_paths) > num_files:
    indices = np.round(np.linspace(0, len(weight_paths) - 1, num_files)).astype(int)
    
    selected_weight_paths = [weight_paths[i] for i in indices]
    weight_paths = selected_weight_paths
    
    print(f"Debug: After uniformly selecting {num_files} files, weight_paths has {len(weight_paths)} entries.")
    if weight_paths:
        print(f"Debug: First uniformly selected weight path: {os.path.basename(weight_paths[0])}")
        print(f"Debug: Last uniformly selected weight path: {os.path.basename(weight_paths[-1])}")
    else:
        print("Debug: No weight paths remaining after uniform selection (this should not happen if initial list had items).")
elif len(weight_paths) > 0 and len(weight_paths) <= num_files:
    print(f"Debug: Total weight files ({len(weight_paths)}) is less than or equal to num_files ({num_files}). Processing all available files.")
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

from scipy.ndimage import binary_dilation, label, distance_transform_edt, sum as ndi_sum

def fill_holes_by_dilate_erode(mask, iterations=2):
    filled_mask = np.zeros_like(mask)
    labels = np.unique(mask)
    labels = labels[labels != 0]  # Skip background

    for label in labels:
        binary = (mask == label)
        # Dilate first to close small holes
        dilated = binary_dilation(binary, iterations=iterations)
        # Erode to restore the shape
        cleaned = binary_erosion(dilated, iterations=iterations)
        filled_mask[cleaned] = label

    return filled_mask
    
def clean_mask(image, mask, border=5, thresh=0.1, sigma=2.8):
    m = (mask > 0)
    struct = np.ones((3, 3, 3), dtype=bool)  # for 3D erosion
    eroded = binary_erosion(m, structure=struct, iterations=border)
    m[~eroded & (image < thresh)] = 0
    lbl, n = label(m)
    if n == 0: return np.zeros_like(mask)
    m = (lbl == 1 + np.argmax(ndi_sum(m, lbl, index=np.arange(1, n + 1))))
    return (gaussian_filter(m.astype(float), sigma) > 0.5).astype(mask.dtype)
    

def refine_prediction2(crop_img,image, mask, mask2, model,folder,
                       orig_voxsize,
                       new_image_size=(192, 192, 192), margin=0, cube_size=192):
    """
    Refines the segmentation prediction in two steps:
    1. Makes an initial prediction.
    2. Crops the image based on the prediction and runs the model again.
    
    Parameters:
    - crop_img (ndarray): The input image for prediction.
    - mask (ndarray): The binary mask.
    - model: The trained segmentation model.
    - new_image_size (tuple): The new voxel size for resizing (default is (192, 192, 192)).
    - margin (int): The margin to add around the bounding box (default is 10).
    - cube_size (int): The size of the bounding cube (default is 32).
    
    Returns:
    - final_prediction_resized (ndarray): The final refined prediction, resized to match the original input size.
    """

    orig_shape = image.shape
    voxsize = image.geom.voxsize
    folder_path = os.path.join(validation_folder_path, folder)
    # os.makedirs(folder_path, exist_ok=True)
    # affine = np.array(image.geom.vox2world)
    
    # Step 1: Initial Prediction
    # Binarize the mask
    mask.data[mask.data != 0] = 1
    # nib.save(nib.Nifti1Image(mask.astype(np.int32), image.geom.vox2world), os.path.join(folder_path, 'mask.nii.gz'))

    # Compute mask center (using the provided find_bounding_box function)
    ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)
    # print(crop_img.shape)
    
    # Make an initial prediction
    prediction_one_hot = model.predict(crop_img[None, ...], verbose=0)
    initial_prediction = np.argmax(prediction_one_hot, axis=-1)[0]

    labeled, num_components = ndimage.label(initial_prediction > 0)
    largest_mask = labeled == np.argmax(ndimage.sum(initial_prediction > 0, labeled, range(num_components + 1)))
    initial_prediction = ndi.binary_fill_holes(largest_mask)
    initial_prediction = (initial_prediction > 0).astype(np.int32)
    initial_prediction = clean_mask(crop_img, initial_prediction,border=2, thresh=0.1, sigma=0.6)

    new_voxsize =1
    # pred_192_3 = combine_mask_with_dilated_label(prediction_seg.astype(np.int32),initial_prediction, target_label=93, dilation_iters=dial_param)
    pred_192_3 = initial_prediction
    # binarized = (initial_prediction > 0).astype(np.int32) * 2
    # binarized = fill_holes_by_dilate_erode(binarized, iterations=2)
    # binarized = clean_mask(crop_img, binarized,border=10, thresh=0.1, sigma=0.6)
    
    # resized = sf.Volume(binarized).reshape(orig_shape).data
    # nib.save(nib.Nifti1Image(resized, affine), os.path.join(folder_path, 'uninjured_prediction.nii.gz'))


        
    return initial_prediction
        


    
# def refine_prediction2(crop_img, image, mask, mask2, model, folder,
#                        orig_voxsize, suffix="",
#                        new_image_size=(192, 192, 192), margin=0, cube_size=128):
#     """
#     Returns the final binary prediction mask (no file saving).
#     """
#     prediction_one_hot = model.predict(crop_img[None, ...], verbose=0)
#     initial_prediction = np.argmax(prediction_one_hot, axis=-1)[0]

#     labeled, num_components = ndi.label(initial_prediction > 0)
#     if num_components == 0:
#         return np.zeros_like(initial_prediction)

#     largest_mask = labeled == np.argmax(ndi.sum(initial_prediction > 0, labeled, range(num_components + 1)))
#     initial_prediction = ndi.binary_fill_holes(largest_mask)
#     initial_prediction = (initial_prediction > 0).astype(np.int32)

#     final_pred = (initial_prediction > 0).astype(np.uint8)
#     return final_pred


# ==== Evaluation loop ====
results = []
# Decide which suffixes to test based on flags
# if t1 and not t2:
test_suffixes = [""]
# elif t2 and not t1:
#     test_suffixes = ["-t2"]
# else:
#     test_suffixes = ["", "-t2"]  # default: test on both if neither or both flags are set

# for weight_path in tqdm(weight_paths, desc="Evaluating Weights"):
    # tf.keras.backend.clear_session()
    
model = load_model_for_weight(latest_weight)
print("#####################",latest_weight)
pig_model = model

dice_scores_for_this_weight = []

for folder in subfolders:
    for suffix in test_suffixes:   # ✅ Only test on selected type (T1/T2)
        folder_path_2 = os.path.join(validation_folder_path, folder)
        image_filename = os.path.join(folder_path_2, f'image.nii.gz')
        mask2_filename = os.path.join(folder_path_2, 'mask.nii.gz')
        mask_filename = os.path.join(folder_path_2, 'mask_drew.nii.gz')

        if not (os.path.isfile(image_filename) and os.path.isfile(mask2_filename) and os.path.isfile(mask_filename)):
            print(f"❌ Missing file in '{folder_path_2}' for suffix '{suffix}' (e.g., '{os.path.basename(image_filename)}') – skipping subject.")
            continue

        try:
            image = sf.load_volume(image_filename)
            affine = np.array(image.geom.vox2world)
            
            crop_img = image.reshape([192, 192, 192, 1])

            mask2 = sf.load_volume(mask2_filename).reshape([192, 192, 192, 1])
            mask = sf.load_volume(mask_filename).reshape([192, 192, 192, 1])
        except Exception as e:
            print(f"Error loading volumes for '{folder_path_2}' with suffix '{suffix}': {e} – skipping subject.")
            continue

        # filename = os.path.join(folder_path_2, 'image.nii.gz')
        # image = sf.load_volume(filename)
        orig_voxsize = image.geom.voxsize
        # crop_img = image.resize(new_voxsize, method="linear").reshape([192, 192, 192, 1])
        crop_img = image.reshape([192, 192, 192, 1])
        orig_shape = image.shape
        
        mask.data[mask.data != 0] = 1
        mask2.data[mask2.data != 0] = 1

        if np.sum(mask.data) < 1000:
            print(f"⚠️ Mask in '{folder_path_2}' with suffix '{suffix}' too small ({int(np.sum(mask.data))} voxels) – skipping subject.")
            continue
        
        # try:
        #     orig_voxsize = image.geom.voxsize
        # except AttributeError:
        #     print(f"Warning: image.geom.voxsize not found for '{folder_path_2}'. Using default (1.0, 1.0, 1.0).")
        #     orig_voxsize = (1.0, 1.0, 1.0)

        # prediction = refine_prediction2(crop_img, image, mask, mask2, pig_model, folder, orig_voxsize)
        prediction = refine_prediction2(crop_img, image, mask, mask2, pig_model, folder,orig_voxsize, new_image_size=(192, 192, 192))

        # mask_flat = mask.data.flatten()
        # prediction_flat = prediction.flatten() > 0

        mask_flat = mask.data.flatten()
        prediction_flat = prediction.flatten()>0
        dice_score = 2 * np.sum(mask_flat * prediction_flat) / (np.sum(mask_flat) + np.sum(prediction_flat))

        prediction = sf.Volume(prediction).reshape(orig_shape).data
        nib.save(nib.Nifti1Image(prediction.astype(np.int32), affine), os.path.join(folder_path_2, 'uninjured_prediction.nii.gz'))

        
        dice_scores_for_this_weight.append(dice_score)
        print(f"Dice for {folder}{suffix}: {dice_score:.4f}")

if dice_scores_for_this_weight:
    mean_dice = np.mean(dice_scores_for_this_weight)
    std_dice = np.std(dice_scores_for_this_weight)
    weight_basename = os.path.basename(latest_weight)
    print(f"✅ Average Dice for {weight_basename}: {mean_dice:.4f} ± {std_dice:.4f}")
    results.append((weight_basename, mean_dice, std_dice))

else:
    print(f"⚠️ No valid dice scores were calculated for {os.path.basename(weight_path)} — this weight will not appear in the final CSV.")

# ==== Final Save to CSV ====
if results:
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(["Weight_File_Name", "Average_Dice"])
        writer.writerow(["Weight_File_Name", "Average_Dice", "Std_Dice"])

        writer.writerows(results)
    print(f"\n✅ Finished. All results saved to: {output_csv}")
else:
    print(f"\n⚠️ No results were generated at all. The CSV file '{output_csv}' was not created or only contains a header.")
    print("Please check previous console output for 'Debug: ERROR', '❌ Missing file', or '⚠️ Mask too small' messages to understand why.")