# from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from neurite_sandbox.tf.models import labels_to_labels
from neurite_sandbox.tf.utils.augment import add_outside_shapes
from neurite.tf.utils.augment import draw_perlin_full

import tensorflow.keras.layers as KL
import voxelmorph as vxm


import argparse
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pathlib
# import surfa as sf
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

from utils import *
from help import *


import pathlib
import json
import nibabel as nib
import numpy as np
import tensorflow as tf
from utils import *
import param_3d
import scipy.ndimage as ndimage
import warnings
from keras import backend as K
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
import sys
from tensorflow.keras.models import load_model
import argparse
from hmrf_em import normalize, kmeans_init, run_hmrf_em
import scipy.ndimage as ndi
from scipy.ndimage import zoom, gaussian_filter
import numpy as np
import os
from utils import *
from help import *

import os
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter, binary_fill_holes, zoom


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-lr','--learning_rate',type=float, default=0.00001, help="learning rate")
parser.add_argument('-zb','--zero_background',type=float, default=0.2, help="zero background")
parser.add_argument('-nc','--nb_conv_per_level',type=int, default=2, help="learning rate")
parser.add_argument('-ie','--initial_epoch',type=int,default=0,help="initial epoch")
parser.add_argument('-sc','--scale',type=float,default=1.0,help="scale")
parser.add_argument('-b','--batch_size',default=8,type=int,help="initial epoch")
parser.add_argument('-m','--num_dims',default=192,type=int,help="number of dims")
parser.add_argument('-k1','--num_brain_classes',default=6,type=int,help="number of dims")
parser.add_argument('-k2','--num_anat_classes',default=6,type=int,help="number of dims")
parser.add_argument('-model', '--model', choices=['gmm','hmrf'], default='192Net')
parser.add_argument('-nt','--num_trainings',default=0,type=int,help="number of dims")

parser.add_argument('-o', '--olfactory', action='store_true', help="Flag to disable number of brain classes")
parser.add_argument('-use_original', '--use_original', action='store_true', help="use original images")
parser.add_argument('-t2', '--t2', action='store_true', help="use t2 images")
parser.add_argument('-t1', '--t1', action='store_true', help="use t1 images")
parser.add_argument('-unique', '--unique', action='store_true', help="unique pigs")
parser.add_argument('-injury', '--injury', action='store_true', help="unique pigs")

args = parser.parse_args()
scaling_factor = 1.0



log_dir = 'logs'
models_dir = 'models'
num_epochs=param_3d.epoch_num
lr=args.learning_rate

print(args.num_trainings)
if args.model=='gmm' and args.num_trainings:
    log_dir += '_gmm_new_'+str(args.num_trainings)+'_'
    models_dir += '_gmm_new_'+str(args.num_trainings)+'_' 
elif args.model=='gmm':
    log_dir += '_gmm_'
    models_dir += '_gmm_'
elif args.model=='hmrf':
    log_dir += '_hmrf_'
    models_dir += '_hmrf_' 

if args.scale != 1:
    scaling_factor = args.scale
    log_dir += '_scale_'+str(args.scale)+'_'
    models_dir += '_scale_'+str(args.scale)+'_' 

if args.t1:
    print("only t1 ########")
    log_dir += 't1_'
    models_dir += 't1_' 
elif args.t2:
    print("only t2 ########")
    log_dir += 't2_'
    models_dir += 't2_' 
elif args.unique:
    print("only unique")
    log_dir += 'unique_'
    models_dir += 'unique_' 

if args.injury:
    print("injury")
    log_dir += 'injury_'
    models_dir += 'injury_' 
    
k1=args.num_brain_classes
k2=args.num_anat_classes

log_dir +=str(args.num_brain_classes)+"_"+str(args.num_anat_classes)
models_dir +=str(args.num_brain_classes)+"_"+str(args.num_anat_classes)

if args.num_dims!=192:
    log_dir +="_"+str(args.num_dims)
    models_dir +="_"+str(args.num_dims)

if args.olfactory:
    log_dir +="_olfactory"
    models_dir +="_olfactory"

if args.use_original:
    log_dir +="_orig"
    models_dir +="_orig"
    
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.95, patience=20, verbose=1, min_lr=1e-7)

weight_files = glob.glob(os.path.join(models_dir, 'weights_*.h5'))
if weight_files:
    # sort by filename string itself → works because YYYY-MM-DD_HH-MM is lexicographic
    latest_weight = sorted(weight_files)[-1]
    checkpoint_path = latest_weight
    print(f"Loading latest weights: {checkpoint_path}")
else:
    checkpoint_path = os.path.join(models_dir, 'weights_epoch_0.h5')
    print("No checkpoint found, starting fresh.")

weights_saver = PeriodicWeightsSaver(filepath=models_dir, save_freq=1)  # Save weights every 100 epochs


early_stopping_callback = EarlyStoppingByLossVal(monitor='loss', value=1e-4, verbose=1)


# Assuming log_dir and models_dir are already defined in your script
TB_callback = CustomTensorBoard(
    base_log_dir=log_dir,
    models_dir=models_dir
)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# path = "/cbica/home/dadashkj/uiuc_pig_brain_atlas/v2.1_12wk_Atlas/Combined_Maps_12wk/Combined_thr50_12wk.nii"
# pig_brain_map = [sf.load_volume(str(path)).resize(0.7).reshape([param_3d.img_size_192,]*3).data]

# path = "/cbica/home/dadashkj/uiuc_pig_brain_atlas/v2.1_12wk_Atlas/Combined_Maps_12wk/Combined_thr50_12wk.nii"
# pig_brain = sf.load_volume(str(path)).resize(1.2).reshape([args.num_dims,]*3).data
# pig_brain = extend_label_map_with_surfa(pig_brain,scale_factor=80)
# pig_brain = dilate_label_map(pig_brain)
# pig_brain_map = [pig_brain]


def mask_bg_near_fg(fg, bg, dilation_iter=5):
    d_iter = tf.random.uniform([], minval=dilation_iter-1, maxval=dilation_iter+1, dtype=tf.int32)
    k = 2 * d_iter + 1
    fg_mask = tf.cast(fg > 0, tf.float32)
    fg_mask = tf.reshape(fg_mask, [1, *fg_mask.shape, 1])
    fg_mask = tf.nn.max_pool3d(fg_mask, ksize=[1, k, k, k, 1], strides=[1, 1, 1, 1, 1], padding='SAME')
    fg_mask = tf.squeeze(fg_mask > 0)

    bg_masked = tf.where(fg_mask, bg, tf.zeros_like(bg[0, ..., 0]))
    result = tf.where(fg > 0, fg, bg_masked)
    return result

def soften_labels_via_gaussian(label_map, sigma=1):
    n_classes = int(label_map.max()) + 1
    smoothed_probs = np.zeros((n_classes,) + label_map.shape)

    for c in range(n_classes):
        class_mask = (label_map == c).astype(float)
        smoothed_probs[c] = gaussian_filter(class_mask, sigma=sigma)

    softened = np.argmax(smoothed_probs, axis=0)
    return softened
    
from scipy.ndimage import zoom, gaussian_filter, binary_fill_holes
from hmrf_em import normalize, kmeans_init, run_hmrf_em
import numpy as np
import os
from utils import *
from help import *

# def build_hmrf_label_map(k1=6, k2=6):
#     # n_classes = k1
#     beta = 1.0
#     max_iter = 10
#     sigma = 1.0  # smoothing before clustering

    
#     folders_path = [
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/template/",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-T2",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3day",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3day-T2",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-1month",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-1month-T2",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3month",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3month-T2",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-6month",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-6month-T2",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/79",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/79-T2",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/106-6month/",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/106-6month-T2/",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/93",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/93-T2"
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/75",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/78",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/82",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/101"
#     ]
    
#     if args.t1:
#         folders_path = [
#             "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/template/",
#             "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3day",
#             "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-1month",
#             "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3month",
#             "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-6month",
#             "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/79",
#             "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/106-6month/",
#             "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/93",
#             "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/75",
#             "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/78",
#             "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/82",
#             "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/101"
#         ]

#     # scaling_factor = 0.7
#     predicted_anat_labels = []

#     def load_volume_file(folder_path, file_name):
#         for ext in ['.nii.gz', '.nii']:
#             file_path = os.path.join(folder_path, file_name + ext)
#             print(f"Checking: {repr(file_path)}")
#             if os.path.exists(file_path):
#                 return sf.load_volume(file_path).reshape([param_3d.img_size_256] * 3).data
#         raise FileNotFoundError(f"{file_name} not found in {folder_path}")

#     for folder_path in folders_path:
#         pig_anat = load_volume_file(folder_path, 'anat')
#         pig_brain_mask = load_volume_file(folder_path, 'anat_brain_olfactory_mask')
#         pig_brain_mask = (pig_brain_mask > 0).astype(np.uint8)

#         # Preprocess
#         pig_anat = gaussian_filter(pig_anat, sigma=sigma)
#         pig_brain_mask = binary_fill_holes(pig_brain_mask)

#         pig_anat = sf.Volume(zoom(pig_anat, scaling_factor, order=1)).reshape((256,) * 3).data
#         pig_brain_mask = sf.Volume(zoom(pig_brain_mask, scaling_factor, order=1)).reshape((256,) * 3).data
#         pig_brain_mask = binary_fill_holes(pig_brain_mask)

#         # Brain region HMRF
#         pig_brain = pig_anat * pig_brain_mask
#         pig_brain = normalize(pig_brain.astype(np.float32))
#         init_brain_labels = kmeans_init(pig_brain, pig_brain_mask, n_classes=k1)
#         brain_seg = run_hmrf_em(pig_brain, pig_brain_mask, init_brain_labels, n_classes=k1, beta=beta, max_iter=max_iter)
#         brain_seg = fill_holes_per_class(brain_seg)
#         brain_seg = soften_labels_via_gaussian(brain_seg,sigma=sigma)
        
#         # Non-brain region HMRF
#         pig_skull = pig_anat.copy()
#         pig_skull[pig_brain_mask == 1] = 0
#         skull_mask = (pig_brain_mask == 0).astype(np.uint8)
#         pig_skull = normalize(pig_skull.astype(np.float32))
#         init_skull_labels = kmeans_init(pig_skull, skull_mask, n_classes=k2)
#         skull_seg = run_hmrf_em(pig_skull, skull_mask, init_skull_labels, n_classes=k2, beta=beta, max_iter=max_iter)

#         skull_seg = fill_holes_per_class(skull_seg)
#         skull_seg = soften_labels_via_gaussian(skull_seg,sigma=sigma)
    

#         # Combine brain and non-brain
#         skull_seg[pig_brain_mask == 1] = 0
#         skull_seg = shift_non_zero_elements(skull_seg, k1)
#         final_seg = np.where(brain_seg > 0, brain_seg, skull_seg)

#         # Final resizing
#         zoomed_predicted_anat_labels = sf.Volume(final_seg).reshape([args.num_dims,] * 3)
#         predicted_anat_labels.append(zoomed_predicted_anat_labels)

#     return predicted_anat_labels

import re
from collections import defaultdict

def extract_pig_id(folder_path):
    if "template" in folder_path:
        return "81"
    match = re.search(r"/(\d+)", folder_path)
    return match.group(1) if match else folder_path

import os
import re
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter, binary_fill_holes
import numpy as np
import surfa as sf
import param_3d
from utils import shift_non_zero_elements

# def build_gmm_label_map(k1=5, k2=6):
#     k1=5

#     injury_keywords = ["3day", "1month", "3month", "6month"]

#     all_folders = [
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-pre",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-pre-T2",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3day",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3day-T2",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-1month",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-1month-T2",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3month",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3month-T2",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-6month",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-6month-T2",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/79",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/79-T2",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/93",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/93-T2",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/75",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/78",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/82",
#         "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/101"
#     ]

#     def load_volume(folder, name):
#         for ext in ['.nii.gz', '.nii']:
#             path = os.path.join(folder, name + ext)
#             if os.path.exists(path):
#                 return sf.load_volume(path).reshape((args.num_dims,) * 3).data
#         raise FileNotFoundError(f"Missing {name} in {folder}")

#     if not args.injury:
#         all_folders = [
#             f for f in all_folders
#             if not any(injury_kw in f for injury_kw in injury_keywords)
#         ]
#     # Filter by --t1 or --t2
#     if args.t1:
#         folders_path = [f for f in all_folders if "-T2" not in f]
#     elif args.t2:
#         folders_path = [f for f in all_folders if "-T2" in f]
#     else:
#         folders_path = all_folders

#     # Unique pig filtering
#     if args.unique:
#         grouped = defaultdict(dict)
#         for folder in folders_path:
#             basename = os.path.basename(folder)

#             # Normalize pig ID and base ID
#             if "template" in folder:
#                 pig_id = "81"
#                 base_id = "template"
#             else:
#                 match = re.search(r'/(\d+)(-[^/]*)?', folder)
#                 pig_id = match.group(1) if match else basename
#                 base_id = basename.replace("-T2", "")

#             if basename.endswith("-T2"):
#                 grouped[base_id]["t2"] = folder
#             else:
#                 grouped[base_id]["t1"] = folder

#         folders_path = []
#         pigs_used = set()

#         # Add template (pig 81) first if it exists
#         if "template" in grouped and len(pigs_used) < args.num_trainings:
#             folders_path.append(grouped["template"]["t1"])
#             if "t2" in grouped["template"]:
#                 folders_path.append(grouped["template"]["t2"])
#             pigs_used.add("81")

#         for base, paths in grouped.items():
#             match = re.match(r"(\d+)", base)
#             pig_id = match.group(1) if match else base

#             if pig_id in pigs_used:
#                 continue
#             if len(pigs_used) >= args.num_trainings:
#                 break

#             if "t1" in paths:
#                 folders_path.append(paths["t1"])
#             if "t2" in paths:
#                 folders_path.append(paths["t2"])
#             pigs_used.add(pig_id)



#     if args.num_trainings > 0:
#         folders_path = folders_path[:args.num_trainings]

        
#     predicted_anat_labels = []
#     sigma = 0.8
#     i = 0

#     while i < len(folders_path):
#         t1_folder = folders_path[i]
#         t2_folder = None

#         if not args.t1 and not args.t2:
#             if i + 1 < len(folders_path):
#                 next_folder = folders_path[i + 1]
#                 if "-T2" in next_folder and t1_folder.replace("-T2", "") in next_folder:
#                     t2_folder = next_folder

#         has_t2 = t2_folder is not None

#         anat_t1 = load_volume(t1_folder, "anat")
#         mask = load_volume(t1_folder, "anat_brain_olfactory_mask") > 0
#         mask = binary_fill_holes(mask).astype(np.uint8)

#         anat_t1 = gaussian_filter(anat_t1, sigma)

#         if has_t2:
#             anat_t2 = load_volume(t2_folder, "anat")
#             anat_t2 = gaussian_filter(anat_t2, sigma)
#             anat_combined = np.stack([anat_t1, anat_t2], axis=-1)
#             brain_data = anat_combined[mask == 1].reshape(-1, 2)
#             non_brain_data = anat_combined[mask == 0].reshape(-1, 2)
#         else:
#             brain_data = anat_t1[mask == 1].reshape(-1, 1)
    
#             # Exclude air from non-brain
#             non_brain_mask = (mask == 0) & (anat_t1 > 0)
#             non_brain_data = anat_t1[non_brain_mask].reshape(-1, 1)

#         # Fit GMM
#         gmm_brain = GaussianMixture(n_components=k1).fit(brain_data)
#         gmm_non_brain = GaussianMixture(n_components=k2).fit(non_brain_data)
    
#         # Prepare label volume
#         flat_shape = np.prod(mask.shape)
#         full_seg = np.zeros(flat_shape, dtype=int)
    
#         # Brain label assignment
#         brain_idx = mask.flatten() == 1
#         brain_pred = gmm_brain.predict(brain_data)
#         full_seg[brain_idx] = brain_pred + 1  # labels: 1 to k1
    
#         # Non-brain label assignment (excluding zero background)
#         anat_flat = anat_t1.flatten()
#         non_brain_mask_flat = (mask.flatten() == 0) & (anat_flat > 0)
#         non_brain_pred = gmm_non_brain.predict(non_brain_data)
#         full_seg[non_brain_mask_flat] = non_brain_pred + k1 + 1  # labels: k1+1 to k1+k2
    
#         # Final reshape
#         full_seg = full_seg.reshape((args.num_dims,) * 3)
#         full_seg = shift_non_zero_elements(full_seg, 1)

#         predicted_anat_labels.append(sf.Volume(full_seg).reshape((args.num_dims,) * 3))


#         i += 2 if has_t2 else 1

#     return predicted_anat_labels

def build_gmm_label_map(k1=5, k2=6):
    k1 = 5  # override, as in your original

    injury_keywords = ["3day", "1month", "3month", "6month"]

    all_folders = [
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-pre",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-pre-T2",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3day",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3day-T2",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-1month",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-1month-T2",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3month",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3month-T2",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-6month",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-6month-T2",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/79",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/79-T2",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/93",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/93-T2",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/75",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/78",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/82",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/101"
    ]

    def load_volume(folder, name):
        for ext in ('.nii.gz', '.nii'):
            path = os.path.join(folder, name + ext)
            if os.path.exists(path):
                vol = sf.load_volume(path)
                if args.num_dims == 128:  # resample for 128 pipeline
                    vol = vol.resize(scaling_factor)
                vol = vol.reshape((192,) * 3)
                return vol.data
        raise FileNotFoundError(f"Missing {name} in {folder}")

    # --- Folder filtering ---
    if not args.injury:
        all_folders = [f for f in all_folders if not any(kw in f for kw in injury_keywords)]
    if args.t1:
        folders_path = [f for f in all_folders if "-T2" not in f]
    elif args.t2:
        folders_path = [f for f in all_folders if "-T2" in f]
    else:
        folders_path = all_folders

    # --- Unique pig filtering ---
    if args.unique:
        grouped = defaultdict(dict)
        for folder in folders_path:
            basename = os.path.basename(folder)
            if "template" in folder:
                pig_id, base_id = "81", "template"
            else:
                match = re.search(r'/(\d+)(-[^/]*)?', folder)
                pig_id = match.group(1) if match else basename
                base_id = basename.replace("-T2", "")
            if basename.endswith("-T2"):
                grouped[base_id]["t2"] = folder
            else:
                grouped[base_id]["t1"] = folder

        folders_path, pigs_used = [], set()
        if "template" in grouped and len(pigs_used) < args.num_trainings:
            folders_path.append(grouped["template"]["t1"])
            if "t2" in grouped["template"]:
                folders_path.append(grouped["template"]["t2"])
            pigs_used.add("81")

        for base, paths in grouped.items():
            pig_id = re.match(r"(\d+)", base).group(1) if re.match(r"(\d+)", base) else base
            if pig_id in pigs_used:
                continue
            if len(pigs_used) >= args.num_trainings:
                break
            if "t1" in paths: folders_path.append(paths["t1"])
            if "t2" in paths: folders_path.append(paths["t2"])
            pigs_used.add(pig_id)

    if args.num_trainings > 0:
        folders_path = folders_path[:args.num_trainings]

    # --- Main loop ---
    predicted_anat_labels, sigma, i = [], 0.8, 0
    while i < len(folders_path):
        t1_folder = folders_path[i]
        t2_folder = None
        if not args.t1 and not args.t2 and i + 1 < len(folders_path):
            if "-T2" in folders_path[i + 1] and t1_folder.replace("-T2", "") in folders_path[i + 1]:
                t2_folder = folders_path[i + 1]
        has_t2 = t2_folder is not None

        anat_t1 = load_volume(t1_folder, "anat")
        mask = load_volume(t1_folder, "anat_brain_olfactory_mask") > 0
        mask = binary_fill_holes(mask).astype(np.uint8)
        anat_t1 = gaussian_filter(anat_t1, sigma)

        if has_t2:
            anat_t2 = load_volume(t2_folder, "anat")
            anat_t2 = gaussian_filter(anat_t2, sigma)
            anat_combined = np.stack([anat_t1, anat_t2], axis=-1)
            brain_mask = mask == 1
            non_brain_mask = (mask == 0) & (anat_t1 > 0)
            brain_data = anat_combined[brain_mask].reshape(-1, 2)
            non_brain_data = anat_combined[non_brain_mask].reshape(-1, 2)
        else:
            brain_mask = mask == 1
            non_brain_mask = (mask == 0) & (anat_t1 > 0)
            brain_data = anat_t1[brain_mask].reshape(-1, 1)
            non_brain_data = anat_t1[non_brain_mask].reshape(-1, 1)

        # Fit GMMs
        gmm_brain = GaussianMixture(n_components=k1).fit(brain_data)
        gmm_non_brain = GaussianMixture(n_components=k2).fit(non_brain_data)

        # Predictions
        brain_pred = gmm_brain.predict(brain_data)
        non_brain_pred = gmm_non_brain.predict(non_brain_data)

        # Assemble full volume using SAME masks
        full_seg = np.zeros(mask.size, dtype=int)
        full_seg[brain_mask.flatten()] = brain_pred + 1
        full_seg[non_brain_mask.flatten()] = non_brain_pred + k1 + 1

        full_seg = full_seg.reshape(mask.shape)
        full_seg = shift_non_zero_elements(full_seg, 1)
        predicted_anat_labels.append(sf.Volume(full_seg).reshape((192,) * 3))

        i += 2 if has_t2 else 1

    return predicted_anat_labels

# predicted_anat_labels=build_gmm_label_map(5,5)
if args.use_original:
    
    
    folders_path = [
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/template/",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-T2",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3day",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3day-T2",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-1month",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-1month-T2",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3month",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3month-T2",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-6month",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-6month-T2",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/79",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/79-T2",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/106-6month/",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/106-6month-T2/",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/93",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/93-T2",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/75",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/78",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/82",
        "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/101"
    ]
    
    if not args.t2:
        folders_path = [
            "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/template/",
            "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3day",
            "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-1month",
            "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-3month",
            "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-6month",
            "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/79",
            "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/106-6month/",
            "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/93",
            "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/75",
            "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/78",
            "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/82",
            "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/101"
        ]
            #     "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/7646",
            # "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/7665",
            # "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/7778"

    image_mask_pairs = load_validation_data_one_hot(folders_path, dim_=args.num_dims)
    pig_gmm_brain_map = generator_from_pairs(image_mask_pairs)
    pig_real_brain_map = pig_gmm_brain_map
elif args.num_dims==param_3d.img_size_128 and args.model=="gmm":
    pig_gmm_brain_map = build_gmm_label_map(6,6)
    config_file= "params_128.json"
elif args.model=="hmrf":
    pig_brain_map = build_hmrf_label_map(6,6)
else:
    pig_brain_map = build_gmm_label_map(k1,6)
    config_file= "params_192.json"


if args.num_dims==param_3d.img_size_96:
    pig_brain_map = pig_gmm_brain_map
    config_file = "params_96.json"
elif args.num_dims==param_3d.img_size_128:
    pig_brain_map = pig_gmm_brain_map
    config_file = "params_128.json"
elif args.num_dims==param_3d.img_size_192:
    config_file = "params_192.json"

# elif args.model=="gmm":
#     pig_brain_map = pig_gmm_brain_map
#     config_file = "params_gmm_192.json"

print("config file")
print(config_file)
with open(config_file, "r") as json_file:
    config = json.load(json_file)

    
gen=generator_brain_window_Net(pig_brain_map,192)

model_pig_config = config["pig_48"]
model_shapes_config = config["shapes"]

model_pig_config["in_shape"]    = [192,192,192]
model_shapes_config["in_shape"] = [192,192,192]

model_pig_config["in_shape"]=[192,192,192]
model_shapes_config["in_shape"]=[192,192,192]

model3_config = config["labels_to_image_model_48"]
model3_config["labels_out"] = {int(key): value for key, value in model3_config["labels_out"].items()}
model3_config["in_shape"]=[192,192,192]
model_pig = create_model(model_pig_config)
model_shapes = create_model(model_shapes_config)
shapes = draw_shapes_easy(shape = (192,)*3)   

labels_to_image_model = create_model(model3_config)

if __name__ == "__main__":
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    random.seed(3000)
    epsilon =1e-7
    steps_per_epoch = 100
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )

    unet_model = vxm.networks.Unet(inshape=(192,192,192, 1), nb_features=(en, de),
                   nb_conv_per_level=2,
                   final_activation_function='softmax')

    if args.use_original:
        
        input_img = Input(shape=(args.num_dims, args.num_dims, args.num_dims, 1))
        normalized_img = min_max_norm(input_img)
        segmentation = unet_model(normalized_img)
        
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.compile(optimizer=Adam(learning_rate=lr), loss=soft_dice)
        

        if os.path.exists(checkpoint_path):
            print(checkpoint_path)
            combined_model.load_weights(checkpoint_path)
            print("Loaded weights from the checkpoint and continued training.")
        callbacks_list = [TB_callback, weights_saver,reduce_lr]
        combined_model.fit(pig_real_brain_map, epochs=num_epochs, batch_size=1, steps_per_epoch=steps_per_epoch,  callbacks=callbacks_list)
        
    elif args.model=="192Net":
        input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192,1))
        _, fg = model_pig(input_img)
        
        shapes = draw_shapes_easy(shape = (param_3d.img_size_192,)*3,num_label=10)
        
        shapes = tf.squeeze(shapes)
        shapes = tf.cast(shapes, tf.int32)
        
        bones = draw_bones_only(shape = (param_3d.img_size_192,)*3,num_labels=16,num_bones=20)
        bones = tf.cast(bones, tf.int32)
        bones = shift_non_zero_elements(bones,29)
        
        shapes2 = draw_layer_elipses(shape=(param_3d.img_size_192,)*3, num_labels=8, num_shapes=50, sigma=2)
        shapes2 = tf.squeeze(shapes2)
        shapes2 = tf.cast(shapes2, tf.int32)
        shapes2 = shift_non_zero_elements(shapes2,29)  
        
        shapes2 = bones + shapes2 * tf.cast(bones == 0,tf.int32)
        result = fg[0,...,0] + shapes2 * tf.cast(fg[0,...,0] == 0,tf.int32)
        result= result[None,...,None]
    
        
        generated_img , y = labels_to_image_model(result)
        generated_img_norm = min_max_norm(generated_img)
        
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=0.00001))
        
    elif args.model=="gmm" and args.num_dims == param_3d.img_size_96:
        input_img = Input(shape=(param_3d.img_size_96,param_3d.img_size_96,param_3d.img_size_96,1))
        
        _, fg = model_pig(input_img[None,...,None])
        _, bg = model_shapes(input_img[None,...,None])

        result = fg[0,...,0] + bg[0,...,0] * tf.cast(fg[0,...,0] == 0,tf.int32)
        result = result[None,...,None]
        generated_img , y = labels_to_image_model(result)
        s
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=lr))

    elif args.num_dims == param_3d.img_size_128:

        input_img = Input(shape=(192,192,192,1))
        
        _, fg = model_pig(input_img[None,...,None])
        _, bg = model_shapes(input_img[None,...,None])

        result = fg[0,...,0] + bg[0,...,0] * tf.cast(fg[0,...,0] == 0,tf.int32)
        result = result[None,...,None]


        generated_img , y = labels_to_image_model(result)
        generated_img_norm = min_max_norm(generated_img)
        
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=lr))


    elif args.num_dims == param_3d.img_size_192:
        input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192,1))
        
        _, fg = model_pig(input_img[None,...,None])
        _, bg = model_shapes(input_img[None,...,None])

        result = fg[0,...,0] + bg[0,...,0] * tf.cast(fg[0,...,0] == 0,tf.int32)
        result = result[None,...,None]


        generated_img , y = labels_to_image_model(result)
        generated_img_norm = min_max_norm(generated_img)
        
        segmentation = unet_model(generated_img_norm)
        combined_model = Model(inputs=input_img, outputs=segmentation)
        combined_model.add_loss(soft_dice(y, segmentation))
        combined_model.compile(optimizer=Adam(learning_rate=lr))
    
    print(checkpoint_path)
    if os.path.exists(checkpoint_path):    
        combined_model.load_weights(checkpoint_path)
        print("Loaded weights from the checkpoint and continued training.")
    else:
        print("checkpoint not found")
                
    callbacks_list = [TB_callback, weights_saver,reduce_lr]
    combined_model.fit(gen, epochs=num_epochs, batch_size=1, steps_per_epoch=steps_per_epoch,  callbacks=callbacks_list)

    