# from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from neurite_sandbox.tf.models import labels_to_labels
from neurite_sandbox.tf.utils.augment import add_outside_shapes
from neurite.tf.utils.augment import draw_perlin_full
from scipy.ndimage import distance_transform_edt
import scipy.ndimage as ndi
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
from tensorflow.keras.layers import Lambda
from utils import *
from help import *
import os

orig=False
num_trainings=18
t1=False
t2=True
unique=False
injury=True
# results_folder="results_doug"
results_folder="results"

# only_t1_model=True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Check devices
print("Available devices:", tf.config.list_physical_devices())

def get_model_folder():
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
        
def get_pig_model_n(n, k1, k2, t1=False, t2=False, unique=False, injury=False):
    epsilon = 1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x) + epsilon) * 1.0)

    print("🔄 Model is loading...")
    en = [16, 16, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    de = [64, 64, 64, 64, 64, 64, 64, 64, 64, 16, 16, 2]
    input_img = Input(shape=(param_3d.img_size_192,) * 3 + (1,))
    unet_model = vxm.networks.Unet(
        inshape=(param_3d.img_size_192,) * 3 + (1,),
        nb_features=(en, de),
        nb_conv_per_level=2,
        final_activation_function='softmax'
    )

    # # Construct model directory name
    # folder_parts = [f"models_gmm_new_{n}"]
    # if unique:
    #     folder_parts.append("unique")
    # if t1:
    #     folder_parts.append("t1")
    # elif t2:
    #     folder_parts.append("t2")
    # if injury:
    #     folder_parts.append("injury")
    # folder_parts.append(f"{k1}_{k2}")
    # model_dir = "_".join(folder_parts)
    model_dir = get_model_folder()
    debug_raw_glob_paths = glob.glob(os.path.join(model_dir, "weights_*.h5"))
    weight_paths = sorted(debug_raw_glob_paths, key=os.path.getmtime)
    latest_weight = weight_paths[-1] if weight_paths else None
    
    # Find latest weight
    # weight_pattern = os.path.join(model_dir, "weights_epoch_*.h5")
    # latest_weight = max(glob.glob(weight_pattern), key=os.path.getctime, default=None)
    
    print(f"📂 Loaded weight: {latest_weight}")
    if not latest_weight:
        raise FileNotFoundError(f"No weight files found in: {model_dir}")

    # Build and load model
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)

    return combined_model

    
def get_pig_model(k1,k2):
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_"+str(k1)+"_"+str(k2), 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    if t1:
        latest_weight = max(glob.glob(os.path.join("models_gmm_t1_"+str(k1)+"_"+str(k2), 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    elif t2:
        latest_weight = max(glob.glob(os.path.join("models_gmm_t2_"+str(k1)+"_"+str(k2), 'weights_epoch_*.h5')), key=os.path.getctime, default=None)

    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model

def get_pig_hmrf_model(k1,k2):
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_hmrf_"+str(k1)+"_"+str(k2), 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model

def get_pig_model_original(k1,k2):
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_"+str(k1)+"_"+str(k2)+"_orig", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model

def get_pig_model_128_original():
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    input_img = Input(shape=(param_3d.img_size_128,param_3d.img_size_128,param_3d.img_size_128, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_128,param_3d.img_size_128,param_3d.img_size_128, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_6_6_128_orig", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model

def get_pig_model_96_full():
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,7]
    input_img = Input(shape=(param_3d.img_size_96,param_3d.img_size_96,param_3d.img_size_96, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_96,param_3d.img_size_96,param_3d.img_size_96, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_seg_FULL_6_6_96", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model

def get_pig_model_atlas():
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,251]
    input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_seg_atlas_6_6", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model

def get_pig_model_atlas_new():
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,103]
    input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_seg_atlas_6_6_new", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model
    
def get_pig_model_96():
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,3]
    input_img = Input(shape=(param_3d.img_size_96,param_3d.img_size_96,param_3d.img_size_96, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_96,param_3d.img_size_96,param_3d.img_size_96, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_seg_6_6_96", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model

def get_pig_hmrf_model_128():
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    input_img = Input(shape=(param_3d.img_size_128,param_3d.img_size_128,param_3d.img_size_128, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_128,param_3d.img_size_128,param_3d.img_size_128, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_hmrf_6_6_128", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model

def get_pig_model_128():
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    input_img = Input(shape=(param_3d.img_size_128,param_3d.img_size_128,param_3d.img_size_128, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_128,param_3d.img_size_128,param_3d.img_size_128, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_6_6_128", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model
    
def get_pig_model_binary_96():
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    input_img = Input(shape=(param_3d.img_size_96,param_3d.img_size_96,param_3d.img_size_96, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_96,param_3d.img_size_96,param_3d.img_size_96, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_6_6_96", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model

def get_pig_model_binary_128():
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,2]
    input_img = Input(shape=(param_3d.img_size_128,param_3d.img_size_128,param_3d.img_size_128, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_128,param_3d.img_size_128,param_3d.img_size_128, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_6_6_128", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model
    
def get_pig_model_atlas_new():
    epsilon =1e-7
    min_max_norm = Lambda(lambda x: (x - K.min(x)) / (K.max(x) - K.min(x)+ epsilon) * (1.0) )
    
    print("model is loading")
    en = [16 ,16 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64 ,64]
    de = [64 ,64 ,64 ,64, 64 ,64 ,64, 64, 64, 16 ,16 ,103]
    input_img = Input(shape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1))
    unet_model = vxm.networks.Unet(inshape=(param_3d.img_size_192,param_3d.img_size_192,param_3d.img_size_192, 1), nb_features=(en, de),
                       nb_conv_per_level=2,
                       final_activation_function='softmax')
        
    latest_weight = max(glob.glob(os.path.join("models_gmm_seg_atlas_6_6_new", 'weights_epoch_*.h5')), key=os.path.getctime, default=None)
    print(latest_weight)
    generated_img_norm = min_max_norm(input_img)
    segmentation = unet_model(generated_img_norm)
    combined_model = Model(inputs=input_img, outputs=segmentation)
    combined_model.load_weights(latest_weight)
    return combined_model

k1=6
k2=6
validation_folder_path = "/cubic/projects/Pig_TBI/JohnWolf/Protocols/T1_mask"

subfolders = [f.name for f in os.scandir(validation_folder_path) if f.is_dir()]
combined_model = get_pig_model(k1,k2)
# pig_model = get_pig_model(k1,k2)
model_name = "gmm"


if model_name == "hmrf":
    pig_model = get_pig_hmrf_model(k1,k2)
else:
    pig_model = get_pig_model(k1,k2)




combined_model_128 = get_pig_model_128()


atlas=True
combined_model_seg = get_pig_model_atlas_new()

if orig:
    combined_model = get_pig_model_original(k1,k2)
elif num_trainings:
    combined_model = get_pig_model_n(num_trainings,k1,k2)

pig_model=combined_model
dial_param = 3 if atlas else 2


import os
import numpy as np
import surfa as sf
import neurite as ne
from scipy import ndimage
from sklearn.metrics import jaccard_score
from utils import find_bounding_box, find_random_bounding_box, apply_gaussian_smoothing, extract_cube
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage import zoom


import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation

from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.cluster import KMeans
import numpy as np

from scipy.ndimage import binary_erosion, binary_dilation, label
import numpy as np

from sklearn.decomposition import PCA

import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import binary_dilation
import numpy as np
from scipy.ndimage import label

from scipy.ndimage import binary_erosion, binary_dilation, label
from scipy.ndimage import binary_erosion, binary_dilation
from sklearn.cluster import KMeans
import numpy as np

# Define a 3D structuring element (e.g., 3x3x3 cube)
structure = np.ones((3, 3, 3), dtype=bool)


validation_folder_path = "/cubic/projects/Pig_TBI/JohnWolf/Protocols/T1_mask"
validation_folder_path = results_folder

# validation_folder_path = "/gpfs/fs001/cbica/home/broodman/Pig_project"

subfolders = [f.name for f in os.scandir(validation_folder_path) if f.is_dir()]
random.shuffle(subfolders)


# combined_model = get_pig_model(k1,k2)
combined_model_128 = get_pig_model_128()


def majority_vote_binary(masks):
    """
    Combine a list of binary 3D masks into a single majority-voted binary mask.
    
    Parameters:
        masks (List[np.ndarray]): List of 3D numpy arrays (binary masks of shape (Z, Y, X))
    
    Returns:
        np.ndarray: 3D binary mask where each voxel is 1 if majority of masks had 1
    """
    masks_stack = np.stack(masks, axis=0)  # shape: (N, Z, Y, X)
    vote_sum = np.sum(masks_stack, axis=0)
    majority_threshold = len(masks) // 2 + 1  # majority means > N/2
    return (vote_sum >= majority_threshold).astype(np.uint8)
    

def fill_holes_per_class(mask, labels=None):
    filled_mask = np.zeros_like(mask)
    if labels is None:
        labels = np.unique(mask)
        labels = labels[labels != 0]  # skip background

    for label in labels:
        class_mask = (mask == label)
        filled_class = ndi.binary_fill_holes(class_mask)
        filled_mask[filled_class] = label

    return filled_mask

from scipy.ndimage import binary_dilation, binary_erosion

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



import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation

# def kmeans_merge_fg_and_prune(img, fg_mask, n_bg_clusters=5):
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation
from scipy.ndimage import gaussian_filter
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from sklearn.mixture import GaussianMixture



def trim_mask(pred_mask, mask_dtype=np.uint8, proximity=2, remove_top_n=1):
    """
    Keeps the dominant class and nearby-attached small classes (within `proximity` voxels),
    but excludes the top-N largest non-dominant labels.

    Parameters:
        pred_mask (np.ndarray): labeled input mask
        mask_dtype (np.dtype): output data type (e.g., np.uint8)
        proximity (int): voxel distance to include nearby labels
        remove_top_n (int): number of largest non-dominant labels to exclude

    Returns:
        np.ndarray: final mask with original labels preserved
    """
    flat = pred_mask.ravel()
    labels, counts = np.unique(flat[flat > 0], return_counts=True)
    if len(counts) == 0:
        return np.zeros_like(pred_mask, dtype=mask_dtype)

    dominant_label = labels[np.argmax(counts)]
    dominant_mask = (pred_mask == dominant_label)
    expanded = binary_dilation(dominant_mask, iterations=proximity)

    # Sizes of all other labels
    label_sizes = {label: np.sum(pred_mask == label) for label in labels if label != dominant_label}
    sorted_labels = sorted(label_sizes.items(), key=lambda x: x[1], reverse=True)
    excluded_labels = set([lbl for lbl, _ in sorted_labels[:remove_top_n]])

    # Initialize final mask with the dominant region
    final_mask = np.zeros_like(pred_mask, dtype=mask_dtype)
    final_mask[dominant_mask] = dominant_label

    # Include small nearby-attached labels
    for label in labels:
        if label == dominant_label or label in excluded_labels:
            continue
        label_mask = (pred_mask == label)
        if np.any(label_mask & expanded):
            final_mask[label_mask] = label

    print("Dominant label:", dominant_label)
    print("Excluded labels (top {}):".format(remove_top_n), excluded_labels)
    print("Labels in final mask:", np.unique(final_mask))

    return final_mask

import numpy as np
from sklearn.decomposition import PCA

import numpy as np
from scipy.ndimage import binary_dilation



import numpy as np
from scipy.ndimage import label

def extract_labeled_largest_component(labeled_mask):
    """
    Returns a labeled mask where only the region under the largest connected component
    (from the binarized version) is preserved. All other areas are set to 0.

    Parameters:
        labeled_mask (np.ndarray): input mask with integer class labels

    Returns:
        np.ndarray: same shape as input, only largest binary component retained
                    with original labels
    """
    binary_mask = (labeled_mask > 0)

    # Step 2: Label connected components
    connected, num = label(binary_mask)
    if num == 0:
        return np.zeros_like(labeled_mask)

    # Step 3: Find largest component
    sizes = np.bincount(connected.ravel())
    sizes[0] = 0  # background
    largest_label = np.argmax(sizes)
    largest_region = (connected == largest_label)

    # Step 4: Mask original labels using that region
    result = np.where(largest_region, labeled_mask, 0)

    return result

from scipy.ndimage import binary_erosion, distance_transform_edt
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

def prune_mask_by_distance_from_core(mask1, mask2, distance_thresh=10, erosion_iter=2):
    """
    Removes distant voxels in mask1 that are farther than `distance_thresh`
    from the eroded core of mask2.

    Parameters:
        mask1 (np.ndarray): binary mask to prune
        mask2 (np.ndarray): reference mask whose core is trusted
        distance_thresh (int): distance threshold in voxels
        erosion_iter (int): erosion depth to define core from mask2

    Returns:
        pruned_mask (np.ndarray): mask1 cleaned based on distance to core of mask2
    """
    # Step 1: Erode mask2 to get core
    core = binary_erosion(mask2, iterations=erosion_iter)

    # Step 2: Compute distance from core
    distance_from_core = distance_transform_edt(~core)

    # Step 3: Remove mask1 voxels that are too far from core
    pruned_mask = (mask1 > 0) & (distance_from_core <= distance_thresh)
    
    return pruned_mask.astype(np.uint8)

import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt

import numpy as np
from scipy.ndimage import binary_dilation

import numpy as np
from scipy.ndimage import binary_dilation, label, sum as ndi_sum

from scipy.ndimage import binary_dilation, label, distance_transform_edt, sum as ndi_sum


def combine_mask_with_dilated_label(pred_192_2, initial_prediction, target_label=93, dilation_iters=50):
    """
    Combines pred_192_2 and initial_prediction into a binary mask.
    Uses binarized pred_192_2 in the largest dilated region around label==target_label,
    and initial_prediction elsewhere.

    Parameters:
        pred_192_2: np.ndarray, multi-class segmentation
        initial_prediction: np.ndarray, binary mask
        target_label: int, label to use as trust anchor (default: 93)
        dilation_iters: int, how many voxels to dilate label region

    Returns:
        final_binary_mask: np.ndarray, 0 for background, 2 for foreground
    """

    # Step 1: Find label==target_label region
    label_region = (pred_192_2 == target_label)
    if not np.any(label_region):
        print(f"Warning: Label {target_label} not found. Using initial_prediction everywhere.")
        return (initial_prediction > 0).astype(np.int32) * 2

    # Step 2: Dilate label region
    dilated_region = binary_dilation(label_region, iterations=dilation_iters)

    # Step 3: Extract largest connected component from dilated region
    labeled_components, num_components = label(dilated_region)
    if num_components == 0:
        print(f"Warning: No connected components after dilation. Using initial_prediction everywhere.")
        return (initial_prediction > 0).astype(np.int32) * 2

    component_sizes = ndi_sum(dilated_region, labeled_components, index=np.arange(1, num_components + 1))
    largest_component_label = np.argmax(component_sizes) + 1  # labels start from 1
    trusted_region = (labeled_components == largest_component_label)

    # Step 4: Binarize pred_192_2 → 0 or 2
    mask_binary = (pred_192_2 > 0).astype(np.int32) * 2

    # Step 5: Binarize initial_prediction → 0 or 2
    init_binary = (initial_prediction > 0).astype(np.int32) * 2

    # Step 6: Merge based on trusted region
    final_mask = np.where(trusted_region, mask_binary, init_binary)

    return final_mask

from scipy.ndimage import binary_erosion, label, sum as ndi_sum

from scipy.ndimage import binary_erosion, gaussian_filter, label
from scipy.ndimage import sum as ndi_sum

from scipy.ndimage import binary_erosion, gaussian_filter, label
from scipy.ndimage import sum as ndi_sum
import numpy as np

def clean_mask(image, mask, border=5, thresh=0.1, sigma=2.8):
    m = (mask > 0)
    struct = np.ones((3, 3, 3), dtype=bool)  # for 3D erosion
    eroded = binary_erosion(m, structure=struct, iterations=border)
    m[~eroded & (image < thresh)] = 0
    lbl, n = label(m)
    if n == 0: return np.zeros_like(mask)
    m = (lbl == 1 + np.argmax(ndi_sum(m, lbl, index=np.arange(1, n + 1))))
    return (gaussian_filter(m.astype(float), sigma) > 0.5).astype(mask.dtype)



def refine_prediction1(crop_img,image, mask, mask2, model, model_128,model_hmrf_128, model_96, folder, new_image_size=(192, 192, 192), margin=0, cube_size=128):
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

    new_voxsize =1
    orig_shape = image.shape
    voxsize = image.geom.voxsize
    folder_path = os.path.join(results_folder, folder)
    os.makedirs(folder_path, exist_ok=True)
    affine = np.array(image.geom.vox2world)
    crop_img = image.resize(new_voxsize, method="linear").reshape([192, 192, 192, 1])
    # nib.save(nib.Nifti1Image(image.data, affine), os.path.join(folder_path, 'image.nii.gz'))
    
    # Step 1: Initial Prediction
    # Binarize the mask
    mask.data[mask.data != 0] = 1
    # nib.save(nib.Nifti1Image(mask.astype(np.int32), image.geom.vox2world), os.path.join(folder_path, 'mask.nii.gz'))

    # Compute mask center (using the provided find_bounding_box function)
    ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)
    print(crop_img.shape)
    
    # Make an initial prediction
    prediction_one_hot = model.predict(crop_img[None, ...], verbose=0)
    initial_prediction = np.argmax(prediction_one_hot, axis=-1)[0]
    ne.plot.volume3D(crop_img, slice_nos=ms)
    print("orig shape",orig_shape)
    print("Initial Prediction Result:")

    labeled, num_components = ndimage.label(initial_prediction > 0)
    largest_mask = labeled == np.argmax(ndimage.sum(initial_prediction > 0, labeled, range(num_components + 1)))
    initial_prediction = ndi.binary_fill_holes(largest_mask)
    initial_prediction = (initial_prediction > 0).astype(np.int32)
    initial_prediction = clean_mask(initial_prediction, initial_prediction,border=2, thresh=0.1, sigma=0.6)

    initial_prediction_dial = binary_dilation(initial_prediction, structure=structure, iterations=dial_param)
    crop_img = crop_img*(initial_prediction_dial>0)


    # crop_img = crop_img*(initial_prediction>0)
    zoom_in_factor = 1
    cube_zoomed = zoom(crop_img, zoom=zoom_in_factor, order=1)  # linear interpolation
    cube_zoomed = sf.Volume(cube_zoomed).reshape((192,192,192))
    # ne.plot.volume3D(cube_zoomed)
    # ne.plot.volume3D(cube_zoomed, slice_nos=ms)
    prediction_cropped_one_hot = combined_model_seg.predict(cube_zoomed[None, ..., None], verbose=0)
    pred_192 = np.argmax(prediction_cropped_one_hot, axis=-1)[0]
    zoom_out_factor = 1
    pred_96_zoomed_out = zoom(pred_192, zoom=zoom_out_factor, order=0)
    print("################## segmentation")
    ne.plot.volume3D(pred_96_zoomed_out,cmaps=['tab20c'])
    pred_192_2 = sf.Volume(pred_96_zoomed_out).reshape((192,192,192)).data
    from scipy.ndimage import binary_erosion, distance_transform_edt


    # pred_192_3 = combine_mask_with_dilated_label2(pred_192_2.astype(np.int32),initial_prediction , prediction_128, 
    #                                              target_label=93,near_label=62,
    #                                              dilation_iters=5)
    pred_192_3 = combine_mask_with_dilated_label(pred_192_2.astype(np.int32),initial_prediction, target_label=93, dilation_iters=50)
    # pred_192_3 = majority_vote_binary([pred_192_3,initial_prediction, prediction_128,prediction_hmrf_128])
    pred_192_3 = clean_mask(crop_img, pred_192_3)
    
    # binarized = (pred_192_3 > 0).astype(np.int32) * 2
    # binarized = fill_holes_by_dilate_erode(binarized, iterations=5)
    # resized = sf.Volume(binarized).resize(voxsize, method="nearest").reshape(orig_shape).data
    # # nib.save(nib.Nifti1Image(resized, affine), os.path.join(folder_path, 'third_prediction.nii.gz'))
    

    pred_192_2[pred_192_3==0]=0 
    pred_192_2 = fill_holes_per_class(pred_192_2)
    nib.save(nib.Nifti1Image(sf.Volume(pred_192_2.astype(np.int32)).resize(voxsize,method="nearest").reshape(orig_shape).data, affine), os.path.join(folder_path, 'seg_prediction.nii.gz'))

    
    return sf.Volume(pred_192_2.astype(np.int32)).resize(voxsize,method="nearest").reshape((192,192,192)).data


# # def refine_prediction2(crop_img,image, mask, mask2, model,folder,
# #                        orig_voxsize,
# #                        new_image_size=(192, 192, 192), margin=0, cube_size=128):
# def refine_prediction2(crop_img, image, mask, mask2, model, folder,
#                        orig_voxsize, suffix="",
#                        new_image_size=(192, 192, 192), margin=0, cube_size=128):
#     """
#     Returns the final binary prediction mask (no file saving).
#     """
#     if suffix=="":
#         suffix="t1"
#     if suffix=="-t2":
#         suffix="t2"
#     affine = np.array(image.geom.vox2world)
#     prediction_one_hot = model.predict(crop_img[None, ...], verbose=0)
#     initial_prediction = np.argmax(prediction_one_hot, axis=-1)[0]

#     labeled, num_components = ndi.label(initial_prediction > 0)
#     if num_components == 0:
#         return np.zeros_like(initial_prediction)

#     largest_mask = labeled == np.argmax(ndi.sum(initial_prediction > 0, labeled, range(num_components + 1)))
#     initial_prediction = ndi.binary_fill_holes(largest_mask)
#     initial_prediction = (initial_prediction > 0).astype(np.int32)

#     final_pred = (initial_prediction > 0).astype(np.uint8)

#     if orig:
#         filename = "orig.nii.gz"
#     elif num_trainings:
#         filename = f"{num_trainings}_{suffix}.nii.gz"
#         if unique:
#             filename = f"{num_trainings}_unique_training.nii.gz"
    
#     # Add t1- prefix if requested
#     if t1:
#         filename = f"t1-{filename}"
#     elif t2:
#         filename = f"t2-{filename}"
        
#     nib.save(nib.Nifti1Image(final_pred, affine), os.path.join(folder_path, filename))
#     return final_pred

def refine_prediction2(crop_img, image, mask, mask2, model, folder,
                       orig_voxsize, suffix="",
                       new_image_size=(192, 192, 192), margin=0, cube_size=128):

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

    if suffix=="":
        suffix="t1"
    if suffix=="-t2":
        suffix="t2"
    orig_shape = image.shape
    voxsize = image.geom.voxsize
    folder_path = os.path.join(results_folder, folder)
    os.makedirs(folder_path, exist_ok=True)
    affine = np.array(image.geom.vox2world)
    # nib.save(nib.Nifti1Image(image.data, affine), os.path.join(folder_path, 'image.nii.gz'))
    
    # Step 1: Initial Prediction
    # Binarize the mask
    mask.data[mask.data != 0] = 1
    # nib.save(nib.Nifti1Image(mask.astype(np.int32), image.geom.vox2world), os.path.join(folder_path, 'mask.nii.gz'))

    # Compute mask center (using the provided find_bounding_box function)
    ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)
    print(crop_img.shape)
    
    # Make an initial prediction
    prediction_one_hot = model.predict(crop_img[None, ...], verbose=0)
    initial_prediction = np.argmax(prediction_one_hot, axis=-1)[0]
    ne.plot.volume3D(crop_img, slice_nos=ms)
    print("orig shape",orig_shape)
    print("Initial Prediction Result:")

    labeled, num_components = ndimage.label(initial_prediction > 0)
    largest_mask = labeled == np.argmax(ndimage.sum(initial_prediction > 0, labeled, range(num_components + 1)))
    initial_prediction = ndi.binary_fill_holes(largest_mask)
    initial_prediction = (initial_prediction > 0).astype(np.int32)
    initial_prediction = clean_mask(crop_img, initial_prediction,border=2, thresh=0.1, sigma=0.6)
    
    nib.save(nib.Nifti1Image(sf.Volume(initial_prediction.astype(np.int32))
                             .reshape(orig_shape).data, affine), os.path.join(folder_path, 'initial_prediction.nii.gz'))

    ne.plot.volume3D(initial_prediction, slice_nos=ms)
    print("first step: ",my_hard_dice(mask.data, initial_prediction))

    new_voxsize =1
    # pred_192_3 = combine_mask_with_dilated_label(prediction_seg.astype(np.int32),initial_prediction, target_label=93, dilation_iters=dial_param)
    pred_192_3 = initial_prediction
    binarized = (initial_prediction > 0).astype(np.int32) * 2
    binarized = fill_holes_by_dilate_erode(binarized, iterations=2)
    binarized = clean_mask(crop_img, binarized,border=10, thresh=0.1, sigma=0.6)
    
    resized = sf.Volume(binarized).reshape(orig_shape).data
    # nib.save(nib.Nifti1Image(resized, affine), os.path.join(folder_path, 'third_prediction.nii.gz'))
    # filename = f"third_prediction_{model_name}.nii.gz"

    filename = f"third_prediction_{model_name}{suffix}.nii.gz"
    if orig:
        filename = "orig.nii.gz"
    elif num_trainings:
        filename = f"{num_trainings}_{suffix}.nii.gz"
        if unique:
            filename = f"{num_trainings}_unique_training.nii.gz"
    
    # Add t1- prefix if requested
    if t1:
        filename = f"t1-{filename}"
    elif t2:
        filename = f"t2-{filename}"
    # filename = f"{model_name}_{suffix}.nii.gz"
    nib.save(nib.Nifti1Image(resized, affine), os.path.join(folder_path, filename))


    tp_fp_map = np.zeros_like(pred_192_3, dtype=np.uint8)
    pred_bin = (pred_192_3 > 0).astype(np.int32)
    # Apply rules
    mask2 = mask2.reshape([192, 192, 192, 1])
    mask1 = mask.reshape([192, 192, 192, 1])
    
    tp_fp_map[(mask2 == 1)] = 3
    tp_fp_map[(mask == 1)] = 2

   
    
    tp_fp_map[(pred_bin == 1) & (mask == 0) & (mask2 == 0)] = 4
    tp_fp_map[(pred_bin == 1) & (mask == 1) & (mask2 == 1)] = 1
    resized = sf.Volume(tp_fp_map).reshape(orig_shape).data
    nib.save(nib.Nifti1Image(resized, affine), os.path.join(folder_path, 'tp_fp_map.nii.gz'))

        
    return pred_192_3>0
        

dice_scores = []

new_voxsize = [0.5, 0.5, 0.5]
# new_voxsize = [0.8, 0.8, 0.8]
import numpy as np
for folder in subfolders:
    for suffix in ["", "-t2"]:  # Run for both image.nii.gz and image-t2.nii.gz
        folder_path = os.path.join(validation_folder_path, folder)
        folder_path_2 = os.path.join(results_folder, folder)
        folder_name = os.path.basename(folder_path)

        image_filename = os.path.join(folder_path_2, f'image{suffix}.nii.gz')
        mask2_filename = os.path.join(folder_path_2, 'mask.nii.gz')
        mask_filename = os.path.join(folder_path_2, 'mask_drew.nii.gz')

        if not os.path.isfile(image_filename):
            if suffix == "":
                print(f"Skipping {folder} — image.nii.gz not found.")
            continue

        if not os.path.isfile(mask2_filename) or not os.path.isfile(mask_filename):
            print(f"Skipping {folder} {suffix} — missing masks.")
            continue

        print(f"Running {folder_name}{suffix}")

        # Load image
        image = sf.load_volume(image_filename)
        orig_voxsize = image.geom.voxsize
        crop_img = image.reshape([192, 192, 192, 1])
        orig_shape = image.shape

        # Load masks
        mask2 = sf.load_volume(mask2_filename).reshape(orig_shape)
        affine = np.array(image.geom.vox2world)
        # nib.save(nib.Nifti1Image(mask2.astype(np.int32), affine), mask2_filename)  # overwrite to enforce shape
        mask2 = mask2.reshape([192, 192, 192, 1])

        mask = sf.load_volume(mask_filename).reshape(orig_shape)
        mask = mask.reshape([192, 192, 192, 1])

        # Binarize masks
        mask.data[mask.data != 0] = 1
        mask2.data[mask2.data != 0] = 1

        # Run prediction (this will handle saving too)
        # prediction = refine_prediction2(crop_img, image, mask, mask2, pig_model, folder, orig_voxsize)
        prediction = refine_prediction2(
            crop_img, image, mask, mask2, pig_model, folder, orig_voxsize,
            suffix=suffix,                       # <-- critical
            new_image_size=(192, 192, 192)
        )
        # prediction = refine_prediction2(
        #     crop_img, image, mask, mask2,
        #     pig_model, folder, orig_voxsize,
        #     suffix=suffix,
        #     new_image_size=(192, 192, 192)
        # )

        # Skip small masks
        if np.sum(mask.data) < 1000:
            continue

        # Compute Dice
        mask_flat = mask.data.flatten()
        prediction_flat = prediction.flatten() > 0
        dice_score = 2 * np.sum(mask_flat * prediction_flat) / (np.sum(mask_flat) + np.sum(prediction_flat))
        dice_scores.append(dice_score)

        print(f"Dice coefficient for {folder_name}{suffix}: {dice_score:.4f}")
        
overall_dice = np.mean(dice_scores)
print(f"Overall Dice coefficient: {overall_dice:.4f}")