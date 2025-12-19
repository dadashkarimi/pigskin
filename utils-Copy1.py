import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
import neurite as ne
from neurite_sandbox.tf.models import labels_to_labels
from neurite_sandbox.tf.utils.augment import add_outside_shapes
from neurite_sandbox.tf.utils.utils import morphology_3d

# from neurite_sandbox.tf.losses import dtrans_loss
from neurite.tf.utils.augment import draw_perlin_full
import voxelmorph as vxm
import os
import glob
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
import keras.backend as K
from tensorflow.keras.losses import categorical_crossentropy
from keras import backend as K
import tensorflow as tf
import tensorflow_probability as tfp
import surfa as sf
import math
# import Image
from skimage.util.shape import view_as_windows
from skimage.transform import pyramid_gaussian
import param_3d
import model_3d
from tensorflow.keras.callbacks import Callback
from utils import *
from skimage.measure import label

import scipy.ndimage as ndi
from skimage.measure import regionprops, marching_cubes
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import svd
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import numpy as np
import cv2

def unify_pig_brain_labels(atlas_volume, csv_mapping_path):
    """
    Unify the labels in a pig brain atlas based on old-to-unified label mapping.

    Args:
        atlas_volume (np.ndarray): 3D numpy array representing the atlas (old labels).
        csv_mapping_path (str): Path to the CSV file containing old and new label mappings.

    Returns:
        np.ndarray: New 3D atlas with unified labels.
    """
    # Load the mapping
    df_mapping = pd.read_csv(csv_mapping_path)

    # Build a dictionary: old label -> new unified label
    mapping_dict = dict(zip(df_mapping["old"], df_mapping["new"]))

    # Create a copy to hold the new labels
    new_atlas = np.zeros_like(atlas_volume, dtype=np.int32)

    # Map old labels to new labels
    for old_label, new_label in mapping_dict.items():
        new_atlas[atlas_volume == old_label] = new_label

    return new_atlas



import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.draw import ellipsoid
from skimage.util import random_noise

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from skimage.util import random_noise

import numpy as np
from scipy.ndimage import binary_dilation

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter

import tensorflow as tf
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter

import tensorflow as tf
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev

import tensorflow as tf
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev
import torch

def load_retina_vessels_with_volume(folder_path, shape=(96, 96, 96), max_images=100):
    """
    Load 2D vessel PNGs, resize to fit into 3D volume, and stack them randomly without thickness or dilation.
    """
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    selected = random.sample(files, min(max_images, len(files)))

    volume = np.zeros(shape, dtype=np.uint8)
    depth = shape[0]

    used_slices = set()

    for i, path in enumerate(selected):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (shape[1], shape[2]))
        binary_mask = (img_resized > 40).astype(np.uint8)

        label = random.randint(1, 10) 

        # Pick a random unused slice
        possible_slices = list(set(range(depth)) - used_slices)
        if not possible_slices:
            break  # no more slices available
        z_idx = random.choice(possible_slices)
        used_slices.add(z_idx)

        # Apply label only where volume is 0
        vessel_slice = binary_mask * label
        volume[z_idx] = np.where(volume[z_idx] == 0, vessel_slice, volume[z_idx])

    return tf.convert_to_tensor(volume, dtype=tf.int32)



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


def img2array(img,dim):
     
    if dim == param.img_size_12:    
        if img.size[0] != param.img_size_12 or img.size[1] != param.img_size_12:
            img = img.resize((param.img_size_12,param.img_size_12))
        img = np.asarray(img).astype(np.float32)/255 
    elif dim == param.img_size_24:
        if img.size[0] != param.img_size_24 or img.size[1] != param.img_size_24:
            img = img.resize((param.img_size_24,param.img_size_24))
        img = np.asarray(img).astype(np.float32)/255
    elif dim == param.img_size_48:
        if img.size[0] != param.img_size_48 or img.size[1] != param.img_size_48:
            img = img.resize((param.img_size_48,param.img_size_48))
        img = np.asarray(img).astype(np.float32)/255
    return img

def calib_box(result_box,result,img):
    

    for id_,cid in enumerate(np.argmax(result,axis=1).tolist()):
        s = cid / (len(param.cali_off_x) * len(param.cali_off_y))
        x = cid % (len(param.cali_off_x) * len(param.cali_off_y)) / len(param.cali_off_y)
        y = cid % (len(param.cali_off_x) * len(param.cali_off_y)) % len(param.cali_off_y) 
                
        s = param.cali_scale[s]
        x = param.cali_off_x[x]
        y = param.cali_off_y[y]
    
        
        new_ltx = result_box[id_][0] + x*(result_box[id_][2]-result_box[id_][0])
        new_lty = result_box[id_][1] + y*(result_box[id_][3]-result_box[id_][1])
        new_rbx = new_ltx + s*(result_box[id_][2]-result_box[id_][0])
        new_rby = new_lty + s*(result_box[id_][3]-result_box[id_][1])
        
        result_box[id_][0] = int(max(new_ltx,0))
        result_box[id_][1] = int(max(new_lty,0))
        result_box[id_][2] = int(min(new_rbx,img.size[0]-1))
        result_box[id_][3] = int(min(new_rby,img.size[1]-1))
        result_box[id_][5] = img.crop((result_box[id_][0],result_box[id_][1],result_box[id_][2],result_box[id_][3]))

    return result_box 



def visualize_image_and_mask(img_data, mask_data, slice_idx=None):
    """
    Visualizes a 3D image and its corresponding mask side by side.
    
    Parameters:
    - img_data (numpy array): The 3D numpy array of the image data.
    - mask_data (numpy array): The 3D numpy array of the mask data.
    - slice_idx (int, optional): The slice index to display. If None, middle slice is used.
    """
    # If no slice index is provided, use the middle slice along the 3rd axis (Z-axis)
    if slice_idx is None:
        slice_idx = img_data.shape[2] // 2

    # Create a figure with subplots for image and mask side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the image slice
    axes[0].imshow(img_data[:, :, slice_idx], cmap='gray')
    axes[0].set_title('Image Slice')
    axes[0].axis('off')  # Hide axis ticks for better viewing

    # Plot the mask slice
    axes[1].imshow(mask_data[:, :, slice_idx], cmap='gray')
    axes[1].set_title('Mask Slice')
    axes[1].axis('off')  # Hide axis ticks for better viewing

    # Display the plots
    plt.show()

# Example usage with already loaded image and mask data:
# visualize_image_and_mask(img_data, mask_data)


def NMS(box):
    
    if len(box) == 0:
        return []
    
    #xmin, ymin, xmax, ymax, score, cropped_img, scale
    box.sort(key=lambda x :x[4])
    box.reverse()

    pick = []
    x_min = np.array([box[i][0] for i in range(len(box))],np.float32)
    y_min = np.array([box[i][1] for i in range(len(box))],np.float32)
    x_max = np.array([box[i][2] for i in range(len(box))],np.float32)
    y_max = np.array([box[i][3] for i in range(len(box))],np.float32)

    area = (x_max-x_min)*(y_max-y_min)
    idxs = np.array(range(len(box)))

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x_min[i],x_min[idxs[1:]])
        yy1 = np.maximum(y_min[i],y_min[idxs[1:]])
        xx2 = np.minimum(x_max[i],x_max[idxs[1:]])
        yy2 = np.minimum(y_max[i],y_max[idxs[1:]])

        w = np.maximum(xx2-xx1,0)
        h = np.maximum(yy2-yy1,0)

        overlap = (w*h)/(area[idxs[1:]] + area[i] - w*h)

        idxs = np.delete(idxs, np.concatenate(([0],np.where(((overlap >= 0.5) & (overlap <= 1)))[0]+1)))
    
    return [box[i] for i in pick]

def sliding_window(img, thr, net, input_12_node):

    pyramid = tuple(pyramid_gaussian(img, downscale = param.downscale))
    detected_list = [0 for _ in xrange(len(pyramid))]
    for scale in xrange(param.pyramid_num):
        
        X = pyramid[scale]

        resized = Image.fromarray(np.uint8(X*255)).resize((int(np.shape(X)[1] * float(param.img_size_12)/float(param.face_minimum)), int(np.shape(X)[0]*float(param.img_size_12)/float(param.face_minimum))))
        X = np.asarray(resized).astype(np.float32)/255

        img_row = np.shape(X)[0]
        img_col = np.shape(X)[1]

        X = np.reshape(X,(1,img_row,img_col,param.input_channel))
        
        if img_row < param.img_size_12 or img_col < param.img_size_12:
            break
        
        #predict and get rid of boxes from padding
        win_num_row = math.floor((img_row-param.img_size_12)/param.window_stride)+1
        win_num_col = math.floor((img_col-param.img_size_12)/param.window_stride)+1

        result = net.prediction.eval(feed_dict={input_12_node:X})
        result_row = np.shape(result)[1]
        result_col = np.shape(result)[2]

        result = result[:,\
                int(math.floor((result_row-win_num_row)/2)):int(result_row-math.ceil((result_row-win_num_row)/2)),\
                int(math.floor((result_col-win_num_col)/2)):int(result_col-math.ceil((result_col-win_num_col)/2)),\
                :]

        feature_col = np.shape(result)[2]

        #feature_col: # of predicted window num in width dim
        #win_num_col: # of box(gt)
        assert(feature_col == win_num_col)

        result = np.reshape(result,(-1,1))
        result_id = np.where(result > thr)[0]
        
        #xmin, ymin, xmax, ymax, score
        detected_list_scale = np.zeros((len(result_id),5),np.float32)
        
        detected_list_scale[:,0] = (result_id%feature_col)*param.window_stride
        detected_list_scale[:,1] = np.floor(result_id/feature_col)*param.window_stride
        detected_list_scale[:,2] = np.minimum(detected_list_scale[:,0] + param.img_size_12 - 1, img_col-1)
        detected_list_scale[:,3] = np.minimum(detected_list_scale[:,1] + param.img_size_12 - 1, img_row-1)

        detected_list_scale[:,0] = detected_list_scale[:,0] / (img_col-1) * (img.size[0]-1)
        detected_list_scale[:,1] = detected_list_scale[:,1] / (img_row-1) * (img.size[1]-1)
        detected_list_scale[:,2] = detected_list_scale[:,2] / (img_col-1) * (img.size[0]-1)
        detected_list_scale[:,3] = detected_list_scale[:,3] / (img_row-1) * (img.size[1]-1)
        detected_list_scale[:,4] = result[result_id,0]

        detected_list_scale = detected_list_scale.tolist()
       
        #xmin, ymin, xmax, ymax, score, cropped_img, scale
        detected_list_scale = [elem + [img.crop((int(elem[0]),int(elem[1]),int(elem[2]),int(elem[3]))), scale] for id_,elem in enumerate(detected_list_scale)]
        
        if len(detected_list_scale) > 0:
            detected_list[scale] = detected_list_scale 
            
    detected_list = [elem for elem in detected_list if type(elem) != int]
    result_box = [detected_list[i][j] for i in xrange(len(detected_list)) for j in xrange(len(detected_list[i]))]
    
    return result_box

def dynamic_resize(image, target_width=192):   

    fov = np.multiply(image.shape, image.geom.voxsize)

    new_voxsize = fov / target_width

    new_voxsize = np.max(new_voxsize[:2])  # ignore slice thickness
    return new_voxsize
    
def load_validation_data(validation_folder_path,dim_):
    subfolders = [f.name for f in os.scandir(validation_folder_path) if f.is_dir()]
    image_mask_pairs = []

    for folder in subfolders:
        folder_path = os.path.join(validation_folder_path, folder)
        filename = os.path.join(folder_path,"image.nii.gz")
        mask_filename = os.path.join(folder_path,"manual.nii.gz")
        image = sf.load_volume(filename)
        new_voxsize = [dynamic_resize(image)]*3

        orig_voxsize = image.geom.voxsize
        crop_img = image.resize([orig_voxsize[0],orig_voxsize[1],1], method="linear")
        crop_img = crop_img.resize(new_voxsize, method="linear").reshape([dim_, dim_, dim_])
        crop_data = crop_img.data
        mask = sf.load_volume(mask_filename).resize([orig_voxsize[0],orig_voxsize[1],1
                                                    ], method="linear")
        mask = mask.resize(new_voxsize).reshape([dim_, dim_, dim_, 1])
        mask.data[mask.data != 0] = 1
        mask.data = tf.cast(mask.data, tf.int32)
        image_mask_pairs.append((crop_data,mask.data))
    return image_mask_pairs

def load_validation_data_one_hot(folders_path, dim_):
    image_mask_pairs = []

    for folder_path in folders_path:
        # Determine image file path (.nii.gz or .mgz)
        if os.path.exists(os.path.join(folder_path, "anat.nii.gz")):
            image_path = os.path.join(folder_path, "anat.nii.gz")
        else:
            image_path = os.path.join(folder_path, "anat.nii")

        mask_path = os.path.join(folder_path, "anat_brain_olfactory_mask.nii.gz")

        # Load and resize image
        image = sf.load_volume(image_path)
        orig_voxsize = image.geom.voxsize
        new_voxsize = [dynamic_resize(image)] * 3

        image = image.resize([orig_voxsize[0], orig_voxsize[1], 1], method="linear")
        image = image.resize(new_voxsize).reshape([dim_, dim_, dim_, 1])

        # Load and resize mask
        mask = sf.load_volume(mask_path)
        mask = mask.resize([orig_voxsize[0], orig_voxsize[1], 1], method="linear")
        mask = mask.resize(new_voxsize).reshape([dim_, dim_, dim_, 1])
        mask.data[mask.data != 0] = 1  # binarize

        # Get bounding box and crop both image and mask
        x1, y1, z1, x2, y2, z2 = find_bounding_box(mask, cube_size=dim_)
        print(f"Bounding box: {x1}, {y1}, {z1}, {x2}, {y2}, {z2}")

        crop_img = extract_cube(image.data, x1, y1, z1, x2, y2, z2)
        crop_img = crop_img[..., None]  # shape: (dim_, dim_, dim_, 1)

        crop_mask = extract_cube(mask.data, x1, y1, z1, x2, y2, z2)
        crop_mask = tf.one_hot(crop_mask.astype(np.uint8), depth=2)  # shape: (dim_, dim_, dim_, 2)
        crop_mask = tf.squeeze(crop_mask)  # squeeze extra dim

        image_mask_pairs.append((crop_img.astype(np.float32), crop_mask.numpy().astype(np.float32)))

    return image_mask_pairs

import numpy as np
import tensorflow as tf
from scipy.ndimage import rotate, shift

import numpy as np
from scipy.ndimage import rotate, shift

import tensorflow as tf

def visualize_tensors(tensors, col_wrap=8, col_names=None, title=None, 
                      figsize_unit=2.5, percentiles=(0.5, 99.5), 
                      cmap='gray', axes_off=True, return_fig_axes=False):
    """
    Visualizes a dictionary of tensors as images in a grid, inspired by neurite.py.

    Args:
        tensors (dict): A dictionary where keys are group names (str) and
                        values are lists or 1D iterable of tensors. Each tensor
                        is expected to be an image (2D, or 3D: C H W, or H W C).
                        If a tensor is None, it will be skipped.
        col_wrap (int): Number of columns to wrap images into for each group.
        col_names (list, optional): List of strings for column titles.
                                    Applies only to the first row of the overall plot.
        title (str, optional): Overall title for the plot.
        figsize_unit (float): Base size for each subplot (width and height).
        percentiles (tuple): Tuple (lower_percentile, upper_percentile) for
                             intensity scaling (vmin, vmax) for 2D (grayscale) images.
                             Commonly (0.5, 99.5) to clip outliers.
        cmap (str): Colormap for 2D (grayscale) images. Defaults to 'gray'.
        axes_off (bool): If True, turn off x and y axis ticks and grid.
        return_fig_axes (bool): If True, return the matplotlib Figure and Axes objects.

    Returns:
        tuple: (fig, axes) if return_fig_axes is True, otherwise None.
    """
    if not isinstance(tensors, dict):
        raise TypeError("Input 'tensors' must be a dictionary.")
    
    if not tensors:
        print("No tensor groups provided to visualize.")
        # Create an empty plot if title is given, otherwise just return.
        if title:
            fig, ax = plt.subplots(1, 1, figsize=(figsize_unit, figsize_unit))
            ax.text(0.5, 0.5, 'No data to display', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            if title:
                plt.suptitle(title, fontsize=20)
            plt.tight_layout()
            plt.show()
            if return_fig_axes:
                return fig, ax
        return

    # Determine maximum number of tensors in any group for layout calculation
    max_tensors_in_group = 0
    for grp_tensors in tensors.values():
        if not isinstance(grp_tensors, (list, tuple)):
            grp_tensors = list(grp_tensors) # Try to convert to list if it's an iterable like a generator
        if len(grp_tensors) > max_tensors_in_group:
            max_tensors_in_group = len(grp_tensors)

    num_groups = len(tensors)
    num_cols = col_wrap
    rows_per_group = math.ceil(max_tensors_in_group / num_cols)
    num_rows = rows_per_group * num_groups

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize_unit * num_cols, figsize_unit * num_rows))

    # Ensure axes is always a 2D array, even for 1x1, 1xN, or Nx1 cases
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, num_cols)
    elif num_cols == 1:
        axes = axes.reshape(num_rows, 1)

    lower_perc, upper_perc = percentiles

    # Iterate through groups and tensors
    for g, (grp_name, grp_tensors) in enumerate(tensors.items()):
        if not isinstance(grp_tensors, (list, tuple)):
            grp_tensors = list(grp_tensors) # Ensure it's a list for consistent indexing

        for k, tensor in enumerate(grp_tensors):
            current_col = k % num_cols
            # Calculate the overall row for the current tensor
            current_row = g * rows_per_group + (k // num_cols)

            # Skip if the tensor is None or if we're out of bounds (shouldn't happen with correct row/col logic)
            if tensor is None:
                ax = axes[current_row, current_col]
                ax.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='lightgray')
                continue # Move to the next tensor

            ax = axes[current_row, current_col]

            # Convert tensor to numpy array and remove singleton dimensions
            if isinstance(tensor, torch.Tensor):
                x = tensor.detach().cpu().numpy()
            else: # Assume it's already a numpy array or convertible
                x = np.asarray(tensor)
            
            x = x.squeeze() # Remove single-dimensional entries (e.g., (1, H, W) -> (H, W))

            if x.ndim == 2: # Grayscale image (H, W)
                # Dynamic vmin/vmax based on percentiles for robust scaling
                data_min = np.percentile(x, lower_perc)
                data_max = np.percentile(x, upper_perc)
                
                # Handle cases where percentiles might be identical (flat image) or cause issues
                if abs(data_max - data_min) < 1e-6:
                    data_min = x.min()
                    data_max = x.max()
                    if abs(data_max - data_min) < 1e-6 and data_max != 0: # Still flat, but non-zero
                        data_min = 0 # Assume 0 is a good lower bound
                        data_max = data_max # Use the actual max
                    elif data_max == 0: # All zeros
                        data_min = 0
                        data_max = 1 # Provide a range for visibility

                ax.imshow(x, vmin=data_min, vmax=data_max, cmap=cmap)

            elif x.ndim == 3: # Color image (C, H, W) or (H, W, C)
                # Heuristic to determine channel axis
                # If the first dimension is 1, 3, or 4, assume C H W and transpose
                if x.shape[0] in [1, 3, 4] and x.shape[1] > 4 and x.shape[2] > 4: # Crude check for C, H, W
                    # If single channel, convert to grayscale 2D
                    if x.shape[0] == 1:
                        data_min = np.percentile(x.squeeze(), lower_perc)
                        data_max = np.percentile(x.squeeze(), upper_perc)
                        if abs(data_max - data_min) < 1e-6:
                            data_min = x.min()
                            data_max = x.max()
                            if abs(data_max - data_min) < 1e-6 and data_max != 0:
                                data_min = 0
                                data_max = data_max
                            elif data_max == 0:
                                data_min = 0
                                data_max = 1
                        ax.imshow(x.squeeze(0), vmin=data_min, vmax=data_max, cmap=cmap)
                    else: # Assume RGB/RGBA
                        # Prefer einops if available for robust rearrangement
                        # if E:
                        #     ax.imshow(E.rearrange(x, 'C H W -> H W C'))
                        # else:
                        ax.imshow(x.transpose(1, 2, 0)) # Fallback: HWC
                elif x.shape[2] in [1, 3, 4] and x.shape[0] > 4 and x.shape[1] > 4: # Assume H W C
                     # If single channel, convert to grayscale 2D
                    if x.shape[2] == 1:
                        data_min = np.percentile(x.squeeze(), lower_perc)
                        data_max = np.percentile(x.squeeze(), upper_perc)
                        if abs(data_max - data_min) < 1e-6:
                            data_min = x.min()
                            data_max = x.max()
                            if abs(data_max - data_min) < 1e-6 and data_max != 0:
                                data_min = 0
                                data_max = data_max
                            elif data_max == 0:
                                data_min = 0
                                data_max = 1
                        ax.imshow(x.squeeze(2), vmin=data_min, vmax=data_max, cmap=cmap)
                    else: # HWC is already suitable for imshow
                        ax.imshow(x)
                else:
                    ax.text(0.5, 0.5, f'Unsupported 3D shape: {x.shape}', 
                            horizontalalignment='center', verticalalignment='center', 
                            transform=ax.transAxes, color='red', fontsize=8)
                    print(f"Warning: Tensor '{grp_name}'[{k}] has an unsupported 3D shape: {x.shape}. Expected (C, H, W) or (H, W, C) where C is 1, 3, or 4.")

            else: # Other dimensions
                ax.text(0.5, 0.5, f'Unsupported shape: {x.shape}', 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=ax.transAxes, color='red', fontsize=8)
                print(f"Warning: Tensor '{grp_name}'[{k}] has an unsupported shape: {x.shape}. Only 2D or 3D images are supported.")

            # Set group label for the first column of each group's first row
            if current_col == 0 and k % col_wrap == 0: # Only set for the first image in each visual row of the group
                ax.set_ylabel(grp_name, fontsize=14)
            
            # Set column names for the very first row of the entire plot
            if col_names is not None and current_row == 0 and current_col < len(col_names):
                ax.set_title(col_names[current_col])

    # Clean up and format all subplots
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axes[i, j]
            # Hide unused subplots
            if i * num_cols + j >= num_rows * num_cols: # This condition might be tricky with varying group lengths
                 fig.delaxes(ax) # Deletes the axes object
            
            # Ensure proper display for all subplots
            if axes_off:
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])

    if title:
        # Adjust layout to make space for suptitle
        plt.suptitle(title, fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Leave space at the top
    else:
        plt.tight_layout()

    plt.show()

    if return_fig_axes:
        return fig, axes

def augment_3d(image, mask):
    """
    Applies fast GPU-friendly 3D augmentations using TensorFlow.
    image: tf.Tensor, shape (D, H, W, 1)
    mask: tf.Tensor, shape (D, H, W, C) with one-hot
    Returns:
        Augmented image and mask
    """

    # Random flip (axes: 0=z, 1=y, 2=x)
    for axis in [0, 1, 2]:
        if tf.random.uniform([]) < 0.5:
            image = tf.reverse(image, axis=[axis])
            mask = tf.reverse(mask, axis=[axis])

    # Brightness adjustment
    if tf.random.uniform([]) < 0.5:
        brightness_factor = tf.random.uniform([], 0.7, 1.3)
        image = tf.clip_by_value(image * brightness_factor, 0.0, 1.0)

    # Gaussian noise
    if tf.random.uniform([]) < 0.5:
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
        image = tf.clip_by_value(image + noise, 0.0, 1.0)

    # Integer voxel shifts (up to ±4)
    for axis in [0, 1, 2]:
        shift_val = tf.random.uniform([], -4, 4, dtype=tf.int32)
        image = tf.roll(image, shift=shift_val, axis=axis)
        mask = tf.roll(mask, shift=shift_val, axis=axis)

    return image, mask



def generator_from_pairs(image_mask_pairs):
    while True:
        img, mask = random.choice(image_mask_pairs)

        # Convert to TensorFlow tensors
        img_tf = tf.convert_to_tensor(img, dtype=tf.float32)
        mask_tf = tf.convert_to_tensor(mask, dtype=tf.float32)

        # Apply GPU-friendly augmentations
        img_aug, mask_aug = augment_3d(img_tf, mask_tf)

        # Convert back to numpy (eager tensors → NumPy arrays)
        yield img_aug[None, ...].numpy(), mask_aug[None, ...].numpy()


# def generator_from_pairs(image_mask_pairs):
#     while True:
#         img, mask = random.choice(image_mask_pairs)  # ✅ use this instead of np.random.choice
#         img_aug, mask_aug = augment_3d(img, mask)
#         yield img_aug[None, ...], mask_aug[None, ...]


# def generator_from_pairs(image_mask_pairs):
#     rand = np.random.default_rng()
#     while True:
#         img, mask = rand.choice(image_mask_pairs)
#         yield img[None, ...], mask[None, ...]  # Add batch dim: (1, D, D, D, 1), (1, D, D, D, C)

# def load_validation_data_one_hot(validation_folder_path,dim_):
#     subfolders = [f.name for f in os.scandir(validation_folder_path) if f.is_dir()]
#     image_mask_pairs = []

#     for folder in subfolders:
#         folder_path = os.path.join(validation_folder_path, folder)
#         filename = os.path.join(folder_path, "image.nii.gz") if os.path.exists(os.path.join(folder_path, "image.nii.gz")) else os.path.join(folder_path, "image.mgz")
#         mask_filename = os.path.join(folder_path,"manual.nii.gz")
#         image = sf.load_volume(filename)
#         new_voxsize = [dynamic_resize(image)]*3
    
        
#         orig_voxsize = image.geom.voxsize
#         image = image.resize([orig_voxsize[0], orig_voxsize[1], 1], method="linear")
#         image = image.resize(new_voxsize).reshape([192, 192, 192, 1])

    
#         mask = sf.load_volume(mask_filename).resize([orig_voxsize[0], orig_voxsize[1], 1], method="linear")
#         mask = mask.resize(new_voxsize).reshape([192, 192, 192, 1])
#         mask.data[mask.data != 0] = 1
    
#         x1, y1, z1, x2, y2, z2 = find_bounding_box(mask,cube_size=dim_)
#         print(x1,y1,z1,x2,y2,z2)
#         crop_img = extract_cube(image.data,x1, y1, z1, x2, y2, z2)
#         crop_img = image.resize([orig_voxsize[0],orig_voxsize[1],1], method="linear")
#         crop_img = crop_img.resize(new_voxsize, method="linear").reshape([dim_, dim_, dim_])
#         mask = extract_cube(mask.data, x1, y1, z1, x2, y2, z2, cube_size=dim_)
        
#         mask = tf.one_hot(mask,depth=2)
#         image_mask_pairs.append((crop_img,mask))
    
        
#     return image_mask_pairs
 
                
def generator_brain_window_Net(label_maps,img_size):
    rand = np.random.default_rng()
    label_maps = np.asarray(label_maps)
    while True:
        fg = rand.choice(label_maps)
        yield fg[None,...,None]

def generator_brain_body(label_maps):
    def create_ellipsoid(center, radii, shape):
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        a, b, c = radii
        cz, cy, cx = center
        
        ellipsoid_mask = ((z - cz) / a) ** 2 + ((y - cy) / b) ** 2 + ((x - cx) / c) ** 2 <= 1
        return ellipsoid_mask

    def create_patch(center, radius, shape):
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        cz, cy, cx = center
        
        patch_mask = (np.sqrt((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) <= radius)
        return patch_mask
    
    def place_large_patches(volume, ellipsoid_mask, num_patches, patch_radius, min_label=9, max_label=16):
        labels = np.random.randint(min_label, max_label + 1, num_patches)
        indices = np.argwhere(ellipsoid_mask)
        
        if len(indices) == 0:
            raise ValueError("Ellipsoid mask is empty. No valid locations for patches.")
        
        selected_indices = indices[np.random.choice(indices.shape[0], num_patches, replace=False)]
        
        for idx, (z, y, x) in enumerate(selected_indices):
            patch_mask = create_patch((z, y, x), patch_radius, volume.shape)
            volume[patch_mask] = labels[idx]
        
        return volume
    
    def attach_ellipsoid_to_volume(volume, min_radius=10, num_patches=10, patch_radius=4):
        non_zero_indices = np.nonzero(volume)
        
        if len(non_zero_indices[0]) == 0:
            raise ValueError("No non-zero elements found in the input volume.")
        
        # Select a random non-zero element
        anchor_idx = np.random.randint(len(non_zero_indices[0]))
        z_anchor, y_anchor, x_anchor = non_zero_indices[0][anchor_idx], non_zero_indices[1][anchor_idx], non_zero_indices[2][anchor_idx]
        
        # Determine ellipsoid semi-axes based on the extent of non-zero elements
        non_zero_extent = np.array([np.max(non_zero_indices[i]) - np.min(non_zero_indices[i]) for i in range(3)])
        min_radius = np.maximum(non_zero_extent, min_radius)
        
        # Randomly shuffle the stretch factors
        base_stretch_factors = np.array([0.5, 0.6, 0.3])
        stretch_factor = np.random.permutation(base_stretch_factors)
        
        # Create ellipsoid centered at the non-zero element
        semi_axes = min_radius * stretch_factor
        ellipsoid = create_ellipsoid((z_anchor, y_anchor, x_anchor), semi_axes, volume.shape)
        
        # Calculate the maximum shift amounts
        max_shift_x = min(volume.shape[2] - (x_anchor + semi_axes[2]), x_anchor)
        max_shift_y = min(volume.shape[1] - (y_anchor + semi_axes[1]), y_anchor)
        max_shift_z = min(volume.shape[0] - (z_anchor + semi_axes[0]), z_anchor)
        
        # Calculate the minimum shift required to ensure non-overlap
        min_shift_x = int(np.ceil(semi_axes[2] * 1.5))  # 1.5 times the semi-axis
        min_shift_y = int(np.ceil(semi_axes[1] * 1.5))
        min_shift_z = int(np.ceil(semi_axes[0] * 1.5))
        
        # Ensure that the shift amount is within bounds
        shift_direction = np.random.choice(['x', 'y', 'z'])
        
        if shift_direction == 'x':
            if min_shift_x > max_shift_x:
                x_shift = 0
            else:
                x_shift = np.random.randint(min_shift_x, max_shift_x + 1)
            y_shift = 0
            z_shift = 0
        elif shift_direction == 'y':
            if min_shift_y > max_shift_y:
                y_shift = 0
            else:
                y_shift = np.random.randint(min_shift_y, max_shift_y + 1)
            x_shift = 0
            z_shift = 0
        else:
            if min_shift_z > max_shift_z:
                z_shift = 0
            else:
                z_shift = np.random.randint(min_shift_z, max_shift_z + 1)
            x_shift = 0
            y_shift = 0
        
        new_center = (
            min(max(z_anchor + z_shift, 0), volume.shape[0] - 1),
            min(max(y_anchor + y_shift, 0), volume.shape[1] - 1),
            min(max(x_anchor + x_shift, 0), volume.shape[2] - 1)
        )
        
        ellipsoid_shifted = create_ellipsoid(new_center, semi_axes, volume.shape)
        
        # Ensure the mask fits within the volume bounds
        ellipsoid_shifted = np.logical_and(ellipsoid_shifted, np.ones(volume.shape, dtype=bool))
        
        updated_volume = np.copy(volume)
        updated_volume[ellipsoid_shifted] = 8  # Set the ellipsoid's value to 8
        
        # Place large patches inside the ellipsoid
        updated_volume = place_large_patches(updated_volume, ellipsoid_shifted, num_patches, patch_radius)
        
        return updated_volume
    
    rand = np.random.default_rng()
    label_maps = np.asarray(label_maps)
    while True:
        fg = rand.choice(label_maps)
        new_fg = attach_ellipsoid_to_volume(fg)
        yield new_fg[None,...,None]
        
def generator_brain_on_off(label_maps, img_size):
    rand = np.random.default_rng()
    label_maps = np.asarray(label_maps)
    
    while True:
        if rand.random() < 0.3:
            # Generate a fully zero fg with the same shape as a label map
            fg = np.zeros_like(label_maps[0])
        else:
            # Randomly select a label map and keep its size
            fg = rand.choice(label_maps)
        
        yield fg[None, ..., None]
        
import tensorflow as tf
from tensorflow.keras.layers import Layer




def find_unique_components(mask1, mask2):
    """
    Find connected components in mask1 that are not present in mask2.

    Parameters:
    mask1 (ndarray): The first 3D binary mask.
    mask2 (ndarray): The second 3D binary mask.

    Returns:
    ndarray: A new mask containing connected components present in mask1 but not in mask2.
    """
    from scipy.ndimage import label
    from skimage.measure import regionprops
    # Ensure masks are binary
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    # Label connected components in both masks
    labeled_mask1, num_features1 = label(mask1)
    labeled_mask2, num_features2 = label(mask2)

    # Initialize a new mask for unique components
    unique_components_mask = np.zeros_like(mask1)

    # Iterate through each region in the first mask
    for region in regionprops(labeled_mask1):
        # Check if any part of the region overlaps with mask2
        if np.all(mask2[labeled_mask1 == region.label] == 0):
            # If no overlap, add the region to the unique components mask
            unique_components_mask[labeled_mask1 == region.label] = 1

    return unique_components_mask


def add_positions(positions,patch_size, mask):

    # Add the position that surrounds non-zero elements of the mask
    positions_list = positions
    x1, y1, z1, x2, y2, z2 = find_bounding_box(mask, cube_size=patch_size)
    positions_list.append((x1, y1, z1, x2, y2, z2))
    
    # Add 10 more positions around the bounding box with a maximum shift of 5 voxels
    for _ in range(10):
        shift_x = np.random.randint(-2, 2)
        shift_y = np.random.randint(-2, 2)
        shift_z = np.random.randint(-2, 2)
        
        new_x1 = x1 + shift_x
        new_y1 = y1 + shift_y
        new_z1 = z1 + shift_z
        
        new_x2 = new_x1 + patch_size
        new_y2 = new_y1 + patch_size
        new_z2 = new_z1 + patch_size
        
        # Check if new coordinates are within bounds
        # if new_x1 < 0 or new_y1 < 0 or new_z1 < 0:
        #     continue
        # if new_x2 > volume_shape[0] or new_y2 > volume_shape[1] or new_z2 > volume_shape[2]:
        #     continue
        
        positions_list.append((new_x1, new_y1, new_z1, new_x2, new_y2, new_z2))
    

    return positions_list

def count_coverage(volume_shape, patch_size, stride):
    coverage = np.zeros(volume_shape, dtype=int)
    
    for x1 in range(0, volume_shape[0] - patch_size + 1, stride):
        for y1 in range(0, volume_shape[1] - patch_size + 1, stride):
            for z1 in range(0, volume_shape[2] - patch_size + 1, stride):
                x2 = x1 + patch_size
                y2 = y1 + patch_size
                z2 = z1 + patch_size
                
                coverage[x1:x2, y1:y2, z1:z2] += 1
    
    return coverage

import random

import random

def generate_position_map(volume_shape, patch_size, stride):
    coverage = count_coverage(volume_shape, patch_size, stride)
    positions = []
    
    # Generate all possible cube positions
    for x1 in range(0, volume_shape[0] - patch_size + 1, stride):
        for y1 in range(0, volume_shape[1] - patch_size + 1, stride):
            for z1 in range(0, volume_shape[2] - patch_size + 1, stride):
                x2 = x1 + patch_size
                y2 = y1 + patch_size
                z2 = z1 + patch_size
                
                # Calculate the weight of this cube based on voxel coverage
                cube_weight = np.sum(coverage[x1:x2, y1:y2, z1:z2])
                
                # Add the position with its weight
                if cube_weight > 0:

                    positions.append(((x1, y1, z1, x2, y2, z2), cube_weight))
    
    # Normalize weights and adjust the position list
    if positions:
        total_weight = sum(weight for _, weight in positions)
        normalized_positions = []
        
        for (pos, weight) in positions:
            # Adjust sample count based on the weight
            sample_count = int(weight / total_weight * len(positions))
            normalized_positions.extend([pos] * sample_count)
    
    # Shuffle positions to mix them up
    np.random.shuffle(normalized_positions)
    
    return normalized_positions
    


# def generate_position_map(volume_shape, patch_size, stride):
#     positions = set()
#     indices = []
#     index = 0

#     # Generate positions based on stride
#     for x1 in range(0, volume_shape[0] - patch_size + 1, stride):
#         for y1 in range(0, volume_shape[1] - patch_size + 1, stride):
#             for z1 in range(0, volume_shape[2] - patch_size + 1, stride):
#                 x2 = x1 + patch_size
#                 y2 = y1 + patch_size
#                 z2 = z1 + patch_size
#                 positions.add((x1, y1, z1, x2, y2, z2))

#     # Convert set to list and generate indices
#     positions = list(positions)

#     return positions


import tensorflow as tf

def generator_validation(image_mask_pairs):                                                                                             
    while True:
        for x, y in image_mask_pairs:
            x = tf.convert_to_tensor(x, dtype=tf.float32)  # Assuming x is your image                                                   
            y = tf.convert_to_tensor(y, dtype=tf.float32)  # Assuming y is your mask                                                    
            yield x[None,...,None] , y[None,...,None]


def generate_position_map_tf(volume_shape, patch_size, stride):
    positions = []
    indices = []

    x_range = tf.range(0, volume_shape[0] - patch_size + 4, stride, dtype=tf.int32)
    y_range = tf.range(0, volume_shape[1] - patch_size + 4, stride, dtype=tf.int32)
    z_range = tf.range(0, volume_shape[2] - patch_size + 4, stride, dtype=tf.int32)

    index = 0
    for x1 in x_range:
        for y1 in y_range:
            for z1 in z_range:
                x2 = x1 + patch_size
                y2 = y1 + patch_size
                z2 = z1 + patch_size
                positions.append((x1, y1, z1, x2, y2, z2))
                indices.append(index)
                index += 1

    # positions = tf.constant(positions, dtype=tf.int32)
    # indices = tf.constant(indices, dtype=tf.int32)

    return positions, indices


def convert_to_binary_vector(pattern_list):
    binary_vector = [1 if pattern is not None else 0 for pattern in pattern_list]
    return binary_vector

def find_bounding_box(mask, cube_size=32, max_dim=192):
    non_zero_coords = np.argwhere(mask)
    
    if non_zero_coords.size == 0:
        # If there are no non-zero elements, choose the cube in the center
        center = np.array(mask.shape) // 2
    else:
        min_coords = non_zero_coords.min(axis=0)
        max_coords = non_zero_coords.max(axis=0)
        center = (min_coords + max_coords) // 2
    
    half_size = cube_size // 2

    # Calculate initial coordinates
    x1 = max(center[0] - half_size, 0)
    y1 = max(center[1] - half_size, 0)
    z1 = max(center[2] - half_size, 0)

    x2 = min(x1 + cube_size, mask.shape[0])
    y2 = min(y1 + cube_size, mask.shape[1])
    z2 = min(z1 + cube_size, mask.shape[2])

    # Adjust if the cube size is smaller than the desired cube size
    if x2 - x1 < cube_size:
        if x1 == 0:
            x2 = min(cube_size, mask.shape[0])
        else:
            x1 = max(x2 - cube_size, 0)

    if y2 - y1 < cube_size:
        if y1 == 0:
            y2 = min(cube_size, mask.shape[1])
        else:
            y1 = max(y2 - cube_size, 0)

    if z2 - z1 < cube_size:
        if z1 == 0:
            z2 = min(cube_size, mask.shape[2])
        else:
            z1 = max(z2 - cube_size, 0)

    return x1, y1, z1, x2, y2, z2
def calibrate_positions(x1, y1, z1, x2, y2, z2, mini_cube_size, max_cube_size):
    x2 = min(x2, max_cube_size)
    y2 = min(y2, max_cube_size)
    z2 = min(z2, max_cube_size)

    # Adjust start coordinates if necessary to maintain the mini cube size
    x1 = max(x2 - mini_cube_size, 0)
    y1 = max(y2 - mini_cube_size, 0)
    z1 = max(z2 - mini_cube_size, 0)

    return x1, y1, z1, x2, y2, z2
    
import numpy as np

import numpy as np

# def find_bounding_box(mask, cube_size=32, max_dim=192):
#     non_zero_coords = np.argwhere(mask)
    
#     if non_zero_coords.size == 0:
#         # If no non-zero elements, return a default bounding box in the center
#         center = np.array(mask.shape) // 2
#         half_size = cube_size // 2
#         x1, y1, z1 = center - half_size
#         x2, y2, z2 = center + half_size
#     else:
#         min_coords = non_zero_coords.min(axis=0)
#         max_coords = non_zero_coords.max(axis=0)
        
#         # Calculate center of non-zero elements
#         center = (min_coords + max_coords) // 2
        
#         # Calculate half size based on cube size
#         half_size = cube_size // 2
        
#         # Calculate bounding box coordinates
#         x1 = max(center[0] - half_size, 0)
#         y1 = max(center[1] - half_size, 0)
#         z1 = max(center[2] - half_size, 0)
#         x2 = min(center[0] + half_size, mask.shape[0])
#         y2 = min(center[1] + half_size, mask.shape[1])
#         z2 = min(center[2] + half_size, mask.shape[2])
        
#         # Adjust if bounding box exceeds max_dim
#         if x2 - x1 > max_dim:
#             x1 = center[0] - max_dim // 2
#             x2 = x1 + max_dim
#         if y2 - y1 > max_dim:
#             y1 = center[1] - max_dim // 2
#             y2 = y1 + max_dim
#         if z2 - z1 > max_dim:
#             z1 = center[2] - max_dim // 2
#             z2 = z1 + max_dim
        
#     return x1, y1, z1, x2, y2, z2





def find_random_bounding_box(input_volume, cube_size=32, margin=8, full_random=False):
    if full_random:
        return find_full_random_bounding_box(input_volume, cube_size)
    
    non_zero_coords = np.argwhere(input_volume)
    if non_zero_coords.size == 0:
        # If no non-zero elements, return the central bounding box
        center = np.array(input_volume.shape) // 2
        half_size = cube_size // 2
        x1, y1, z1 = center - half_size
        x2, y2, z2 = center + half_size
        return x1, y1, z1, x2, y2, z2

    min_coords = np.maximum(non_zero_coords.min(axis=0) - margin, 0)
    max_coords = np.minimum(non_zero_coords.max(axis=0) + margin, np.array(input_volume.shape) - cube_size)
    
    random_coords = [np.random.randint(min_c, max_c + 1) for min_c, max_c in zip(min_coords, max_coords)]
    x1, y1, z1 = random_coords
    x2, y2, z2 = x1 + cube_size, y1 + cube_size, z1 + cube_size

    # Ensure the bounding box is within the input volume size
    if x2 > input_volume.shape[0]:
        x1 = input_volume.shape[0] - cube_size
        x2 = input_volume.shape[0]
    if y2 > input_volume.shape[1]:
        y1 = input_volume.shape[1] - cube_size
        y2 = input_volume.shape[1]
    if z2 > input_volume.shape[2]:
        z1 = input_volume.shape[2] - cube_size
        z2 = input_volume.shape[2]

    return x1, y1, z1, x2, y2, z2


# def find_random_bounding_box(input_volume, cube_size=32, margin=8, full_random=False):
#     if full_random:
#         return find_full_random_bounding_box(input_volume,cube_size)
#     non_zero_coords = np.argwhere(input_volume)
#     min_coords = non_zero_coords.min(axis=0) - margin
#     max_coords = non_zero_coords.max(axis=0) + margin
#     random_coords = [np.random.randint(min_c, max_c - cube_size + 1) for min_c, max_c in zip(min_coords, max_coords)]
#     x1, y1, z1 = random_coords
#     x2, y2, z2 = x1 + cube_size, y1 + cube_size, z1 + cube_size
#     return x1, y1, z1, x2, y2, z2
    
def find_full_random_bounding_box(input_volume, cube_size=32):
    x_max, y_max, z_max = input_volume.shape
    x1 = np.random.randint(0, x_max - cube_size)
    y1 = np.random.randint(0, y_max - cube_size)
    z1 = np.random.randint(0, z_max - cube_size)
    x2 = x1 + cube_size
    y2 = y1 + cube_size
    z2 = z1 + cube_size
    return x1, y1, z1, x2, y2, z2
    
def extract_cube(input_volume, x1, y1, z1, x2, y2, z2, cube_size=32):
    cube_size = x2-x1
    cube = np.zeros((cube_size, cube_size, cube_size))
    x_size, y_size, z_size = x2 - x1, y2 - y1, z2 - z1
    cube[:x_size, :y_size, :z_size] = input_volume[x1:x2, y1:y2, z1:z2]
    return cube

# def extract_cube(input_volume, x1, y1, z1, x2, y2, z2,cube_size=32):
#     cube_size = (x2 - x1, y2 - y1, z2 - z1)
#     cube = np.zeros(cube_size)
#     x_start, x_end = max(0, -x1), min(cube_size[0], input_volume.shape[0] - x1)
#     y_start, y_end = max(0, -y1), min(cube_size[1], input_volume.shape[1] - y1)
#     z_start, z_end = max(0, -z1), min(cube_size[2], input_volume.shape[2] - z1)
#     cube[x_start:x_end, y_start:y_end, z_start:z_end] = input_volume[x1+x_start:x1+x_end, y1+y_start:y1+y_end, z1+z_start:z1+z_end]
#     return cube


# def extract_cube(input_volume, x1, y1, z1, x2, y2, z2, cube_size=32):
#     max_cube_size = min(cube_size, min(x2 - x1, y2 - y1, z2 - z1, 192))
#     cube = np.zeros((max_cube_size, max_cube_size, max_cube_size))
#     x_size, y_size, z_size = min(x2 - x1, max_cube_size), min(y2 - y1, max_cube_size), min(z2 - z1, max_cube_size)
#     cube[:x_size, :y_size, :z_size] = input_volume[x1:x1 + x_size, y1:y1 + y_size, z1:z1 + z_size]
#     return cube

def create_oval_around_fg(fg, margin=4):
    """
    Create an oval shape surrounding the non-zero elements of the foreground.

    Args:
    fg (tf.Tensor): 3D tensor representing the foreground.
    margin (int): Margin to add around the foreground.

    Returns:
    tf.Tensor: 3D tensor representing the oval shape.
    """
    # Get the non-zero coordinates of fg
    non_zero_coords = tf.where(tf.not_equal(fg, 0))
    
    # Calculate the center and radii for the oval
    min_coords = tf.reduce_min(non_zero_coords, axis=0)
    max_coords = tf.reduce_max(non_zero_coords, axis=0)
    center = (min_coords + max_coords) // 2
    radii = (max_coords - min_coords) // 2 + margin

    # Create a grid of coordinates
    grid = tf.stack(tf.meshgrid(tf.range(fg.shape[0]), tf.range(fg.shape[1]), tf.range(fg.shape[2]), indexing='ij'), axis=-1)
    
    # Calculate the distances from the center
    distances = tf.reduce_sum(((tf.cast(grid, tf.float32) - tf.cast(center, tf.float32)) / tf.cast(radii, tf.float32)) ** 2, axis=-1)
    
    # Create an oval shape (ellipsoid)
    oval = distances <= 1

    return tf.cast(oval, tf.uint8)

def weighted_centroid(pred_24):
    # Ensure pred_24 is a numpy array
    pred_24 = np.array(pred_24)
    
    # Get the coordinates of each voxel
    x_coords, y_coords, z_coords = np.indices(pred_24.shape)
    
    # Calculate the weighted sum of coordinates
    total_weight = np.sum(pred_24)
    weighted_sum_x = np.sum(x_coords * pred_24)
    weighted_sum_y = np.sum(y_coords * pred_24)
    weighted_sum_z = np.sum(z_coords * pred_24)
    
    # Compute the weighted centroid
    centroid_x = weighted_sum_x / total_weight
    centroid_y = weighted_sum_y / total_weight
    centroid_z = weighted_sum_z / total_weight
    
    return (centroid_x, centroid_y, centroid_z)
    
import numpy as np

def shift_cube_by_k_voxels(pred_24, k, x1, y1, z1, x2, y2, z2, img_size_24, img_size_48):
    # Get non-zero coordinates
    non_zero_coords = np.argwhere(pred_24 > 0)
    
    # Compute the centroid of non-zero elements
    centroid = non_zero_coords.mean(axis=0).astype(int)
    
    # Calculate shifts based on the centroid relative to the center of the cube
    shifts = (np.sign(centroid - np.array(pred_24.shape) / 2) * k).astype(int)

    # New coordinates after applying the shift
    x1_new = max(0, int(x1 + shifts[0]))
    y1_new = max(0, int(y1 + shifts[1]))
    z1_new = max(0, int(z1 + shifts[2]))

    # Compute the new x2, y2, z2 ensuring the cube remains within the bounds
    x2_new = min(x1_new + img_size_24, img_size_48)
    y2_new = min(y1_new + img_size_24, img_size_48)
    z2_new = min(z1_new + img_size_24, img_size_48)

    # Ensure that the cube size remains consistent
    if x2_new - x1_new != img_size_24:
        x1_new = max(0, x2_new - img_size_24)
    if y2_new - y1_new != img_size_24:
        y1_new = max(0, y2_new - img_size_24)
    if z2_new - z1_new != img_size_24:
        z1_new = max(0, z2_new - img_size_24)

    # Final adjustment to make sure the cube is within bounds
    x2_new = min(x1_new + img_size_24, img_size_48)
    y2_new = min(y1_new + img_size_24, img_size_48)
    z2_new = min(z1_new + img_size_24, img_size_48)

    return x1_new, y1_new, z1_new, x2_new, y2_new, z2_new

    
                    
def create_oval(shape, center, radii):
    grid = tf.stack(tf.meshgrid(tf.range(shape[0], dtype=tf.int32), tf.range(shape[1], dtype=tf.int32), tf.range(shape[2], dtype=tf.int32), indexing='ij'), axis=-1)
    center = tf.reshape(center, (1, 1, 1, -1))  # Reshape center for broadcasting
    radii = tf.reshape(radii, (1, 1, 1, -1))    # Reshape radii for broadcasting
    distances = tf.reduce_sum(((tf.cast(grid, tf.float32) - tf.cast(center, tf.float32)) / tf.cast(radii, tf.float32)) ** 2, axis=-1)
    oval = distances <= 1
    return tf.cast(oval, tf.uint8)

def create_oval_around_fg(fg, margin=10):
    non_zero_coords = tf.where(tf.not_equal(fg, 0))
    min_coords = tf.reduce_min(non_zero_coords, axis=0)
    max_coords = tf.reduce_max(non_zero_coords, axis=0)
    center = (min_coords + max_coords) // 2
    radii = (max_coords - min_coords) // 2 + margin
    return create_oval(fg.shape, center, radii)


def create_smaller_oval_adjacent_to_fg(fg, margin1=30, margin2=1, size_fraction=0.3, shift=20):
    main_oval = create_oval_around_fg(fg, margin1)
    non_zero_coords = tf.where(main_oval)
    min_coords = tf.reduce_min(non_zero_coords, axis=0)
    max_coords = tf.reduce_max(non_zero_coords, axis=0)
    center = tf.cast((min_coords + max_coords) // 2, dtype=tf.int32)
    radii = tf.cast((max_coords - min_coords) // 2 + margin2, tf.float32)
    smaller_radii = tf.cast(radii * size_fraction, dtype=tf.int32)
    
    # Determine the direction to shift the smaller oval
    fg_center = tf.cast(tf.reduce_mean(non_zero_coords, axis=0), dtype=tf.int32)
    shift_direction = tf.constant([shift, 0, 0], dtype=tf.int32)
    
    smaller_center = center + shift_direction
    smaller_oval = create_oval(fg.shape, smaller_center, smaller_radii)
    return main_oval, smaller_oval
    

def create_ellipsoid_around_brain(brain_volume, margin=2, stretch_range=(1.0, 3.0)):
    """
    Create a randomly stretched ellipsoid around the brain volume.

    Args:
        brain_volume (tf.Tensor): 3D tensor of the brain volume.
        margin (int): Additional margin to add around the brain bounding box.
        stretch_range (tuple): Range of factors to stretch the ellipsoid along each axis.

    Returns:
        tf.Tensor: 3D tensor of the ellipsoid mask.
    """
    # Find the bounding box of the brain
    non_zero_indices = tf.where(brain_volume > 0)
    min_z, max_z = tf.reduce_min(non_zero_indices[:, 0]), tf.reduce_max(non_zero_indices[:, 0])
    min_y, max_y = tf.reduce_min(non_zero_indices[:, 1]), tf.reduce_max(non_zero_indices[:, 1])
    min_x, max_x = tf.reduce_min(non_zero_indices[:, 2]), tf.reduce_max(non_zero_indices[:, 2])

    # Calculate the center and radii of the ellipsoid
    center = tf.cast([(max_z + min_z) / 2, (max_y + min_y) / 2, (max_x + min_x) / 2], tf.float32)
    radii = tf.cast([(max_z - min_z) / 2, (max_y - min_y) / 2, (max_x - min_x) / 2], tf.float32) + margin

    # Generate random stretch factors
    random_stretch_factors = tf.random.uniform(shape=[3], minval=stretch_range[0], maxval=stretch_range[1], dtype=tf.float32)
    
    # Apply the random stretch factors to the radii
    radii = radii * random_stretch_factors

    # Create a grid of coordinates
    z = tf.range(brain_volume.shape[0], dtype=tf.float32)
    y = tf.range(brain_volume.shape[1], dtype=tf.float32)
    x = tf.range(brain_volume.shape[2], dtype=tf.float32)
    
    z_grid, y_grid, x_grid = tf.meshgrid(z, y, x, indexing='ij')

    # Calculate the ellipsoid formula
    ellipsoid = ((x_grid - center[2]) ** 2 / radii[2] ** 2 +
                 (y_grid - center[1]) ** 2 / radii[1] ** 2 +
                 (z_grid - center[0]) ** 2 / radii[0] ** 2) <= 1

    return tf.cast(ellipsoid, tf.uint8)
    

def wrap_with_label_tf(volume, thickness=1, iterations=1):
    # Convert volume to float32 for smoother operations
    volume = tf.cast(volume, dtype=tf.float32)
    
    # Create a mask where 1 represents non-zero elements in the volume
    non_zero_mask = tf.cast(tf.not_equal(volume, 0), dtype=tf.float32)
    
    # Define a kernel for convolution
    kernel_size = 2 * thickness + 1  # Ensuring the kernel covers the thickness properly
    kernel = tf.ones((kernel_size, kernel_size, kernel_size, 1, 1), dtype=tf.float32)
    
    # Convolve the non_zero_mask multiple times to create a thicker and smoother ring
    smoothed_mask = non_zero_mask
    for _ in range(iterations):
        smoothed_mask = tf.nn.conv3d(smoothed_mask, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
        smoothed_mask = tf.minimum(smoothed_mask, 1.0)  # Ensure values are within [0, 1]
    
    # Create a mask for the final wrapped volume (excluding original non-zero elements)
    wrapped_mask = tf.cast(tf.logical_and(tf.equal(smoothed_mask, 1), tf.equal(non_zero_mask, 0)), dtype=tf.uint8)
    
    # Combine original volume and wrapped mask
    wrapped_volume = tf.cast(volume, dtype=tf.uint8) + wrapped_mask * 30
    
    return wrapped_volume
    
def extract_fragments(input_volume, positions):
    fragments = []
    for pos in positions:
        x1, y1, z1, x2, y2, z2 = pos
        fragment = input_volume[x1:x2, y1:y2, z1:z2]
        fragments.append(fragment)
    return np.stack(fragments)

def extract_single_fragment(input_volume, position):
    x1, y1, z1, x2, y2, z2 = position
    fragment = input_volume[x1:x2, y1:y2, z1:z2]
    return fragment

def random_select_tensor(bg, result,maxval=2):
    random_val = tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32)
    return tf.cond(tf.equal(random_val, 0), lambda: bg, lambda: result)


def random_selection_layer(bg, result,maxval=2):
    return Lambda(lambda x: random_select_tensor(x[0], x[1],maxval))([bg, result])
    

def apply_gaussian_smoothing(tensor,sigma = 1.0,kernel_size = 3):
    kernel = tf.exp(-0.5 * tf.square(tf.linspace(-1.0, 1.0, kernel_size)) / sigma**2)
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.reshape(kernel, [kernel_size, 1, 1, 1, 1])
    kernel = tf.tile(kernel, [1, kernel_size, 1, 1, 1])
    kernel = tf.tile(kernel, [1, 1, kernel_size, 1, 1])
    return tf.nn.conv3d(tensor, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')


def add_random_blur(image, blur_prob=0.05, sigma=1.0, kernel_size=5):
    # Convert to 3D tensor if needed
    if len(image.shape) == 5:
        image = image[0, ..., 0]  # Shape: (depth, height, width, channels)

    # Apply blurring with probability
    if tf.random.uniform([]) < blur_prob:
        blurred_image = apply_gaussian_smoothing(image, sigma=sigma, kernel_size=kernel_size)
    else:
        blurred_image = image

    # Ensure the image is within original value range if necessary
    # blurred_image = tf.clip_by_value(blurred_image, tf.reduce_min(image), tf.reduce_max(image))
    
    # Add batch and channel dimensions back
    blurred_image = blurred_image = blurred_image[None,...,None]
    
    return blurred_image
    
def shift_non_zero_elements(bg, shift_value):
    non_zero_mask = tf.not_equal(bg, 0)
    shifted_non_zero_elements = tf.where(non_zero_mask, bg + shift_value, bg)
    return shifted_non_zero_elements
    
def generator_brain_label_maps(brain_maps):
    label_maps = np.asarray(brain_maps)
    for fg in label_maps:
        yield fg[None, ..., None]

def create_model(model_config):
    model_config_ = model_config.copy()
    # return model_3d.labels_to_image_new(**model_config_)
    return ne.models.labels_to_image(**model_config_)

def soft_dice(a, b):
    dim = len(a.shape) - 2
    space = list(range(1, dim + 1))

    top = 2 * tf.reduce_sum(a * b, axis=space)
    bot = tf.reduce_sum(a ** 2, axis=space) + tf.reduce_sum(b ** 2, axis=space)
    
    out = tf.divide(top, bot + 1e-6)
    return -tf.reduce_mean(out)
    
def create_window_model(positions, window_size, model_config):
    model_config_ = model_config.copy()
    model_config["window_size"]=window_size
    model_config["positions"]=positions
    return ne.models.labels_to_windows_new(**model_config_)

import tensorflow as tf
from tensorflow.keras.layers import Layer

def mask_bg_near_fg(fg, bg, dilation_iter=8):
    d_iter = tf.random.uniform([], minval=1, maxval=dilation_iter + 1, dtype=tf.int32)
    k = 2 * d_iter + 1
    fg_mask = tf.cast(fg[0, ..., 0] > 0, tf.float32)
    fg_mask = tf.reshape(fg_mask, [1, *fg_mask.shape, 1])
    fg_mask = tf.nn.max_pool3d(fg_mask, ksize=[1, k, k, k, 1], strides=[1, 1, 1, 1, 1], padding='SAME')
    fg_mask = tf.squeeze(fg_mask > 0)

    bg_masked = tf.where(fg_mask, bg[0, ..., 0], tf.zeros_like(bg[0, ..., 0]))
    result = tf.where(fg[0, ..., 0] > 0, fg[0, ..., 0], bg_masked)
    return result


import numpy as np

class GenerateLabelsLayer(tf.keras.layers.Layer):
    def __init__(self, volume_shape, positions, patch_size, stride, **kwargs):
        super(GenerateLabelsLayer, self).__init__(**kwargs)
        self.volume_shape = volume_shape
        self.positions = positions

    def call(self, y):
        labels = []
        # print(y.shape,y)
        brain_indices = np.where(y > 0)
        print(brain_indices.shape)
        for pos in self.positions:
            x1, y1, z1, x2, y2, z2 = pos
            condition = (
                x1 >= brain_indices[0].min() and
                y1 >= brain_indices[1].min() and
                z1 >= brain_indices[2].min() and
                x2 <= brain_indices[0].max() and
                y2 <= brain_indices[1].max() and
                z2 <= brain_indices[2].max()
            )
            labels.append(1 if condition else 0)
        return np.array(labels)




@tf.function
def process_fragment(x1, y1, z1, x2, y2, z2, img_size, y):
    if tf.executing_eagerly():
        fragment = tf.zeros((img_size, img_size, img_size), dtype=tf.float32)
        label = tf.constant(0, dtype=tf.float32)
    else:
        mask = y
        brain_indices = tf.where(mask > 0)
        condition = (
            x1 >= tf.reduce_min(brain_indices[:, 0]) and
            y1 >= tf.reduce_min(brain_indices[:, 1]) and
            z1 >= tf.reduce_min(brain_indices[:, 2]) and
            x2 <= tf.reduce_max(brain_indices[:, 0]) and
            y2 <= tf.reduce_max(brain_indices[:, 1]) and
            z2 <= tf.reduce_max(brain_indices[:, 2])
        )
        label = tf.cond(condition, lambda: tf.constant(1), lambda: tf.constant(0))
    
    return fragment, label


def produce_labels(y_shape, positions):
    labels = tf.zeros(y_shape[1], dtype=tf.int32)
    return labels


def generator_fragments_and_labels(label_map, model_feta, model_shapes, labels_to_image_model, img_size, batch_size):
    input_img = Input(shape=(192, 192, 192, 1))
    _, fg = model_feta(input_img)
    shapes = draw_shapes_easy(shape=(192,) * 3)
    shapes = tf.squeeze(shapes)
    shapes = tf.cast(shapes, tf.uint8)
    _, bg = model_shapes(shapes[None, ..., None])
    bg = bg + 8
    fg_inner = fg[0, ..., 0]
    bg_inner = tf.reshape(bg[0, ..., 0], fg_inner.shape)
    mask = tf.cast(tf.equal(fg_inner, 0), fg_inner.dtype)
    result = fg_inner + bg_inner * mask

    generated_img, y = labels_to_image_model(result[None, ..., None])
    
    if len(generated_img) == 0:
        yield tf.zeros((1, img_size, img_size, img_size, 1), dtype=tf.float32), tf.constant([0], dtype=tf.float32)
    else:
        for img, label in zip(generated_img, y):
            yield img[None, ...], tf.constant([label], dtype=tf.float32)
            
    # for i in range(0, len(positions), batch_size):
    #     fragment_batch = []
    #     label_batch = []
        
    #     for pos in positions[i:i+batch_size]:
    #         x1, y1, z1, x2, y2, z2 = pos
    #         # fragment = generated_img[0, x1:x2, y1:y2, z1:z2, 0]
    #         fragment, label = process_fragment(x1, y1, z1, x2, y2, z2, img_size, y)
    #         # if tf.executing_eagerly():
    #         #     fragment = tf.zeros((img_size, img_size, img_size), dtype=tf.float32)
    #         #     label = tf.constant(0,dtype=tf.float32)
    #         # else:
    #         #     mask = y
    #         #     brain_indices = tf.where(mask > 0)
    #         #     print("#####",tf.reduce_sum(brain_indices))
    #         #     condition = (
    #         #         x1 >= tf.reduce_min(brain_indices[:, 0]) and
    #         #         y1 >= tf.reduce_min(brain_indices[:, 1]) and
    #         #         z1 >= tf.reduce_min(brain_indices[:, 2]) and
    #         #         x2 <= tf.reduce_max(brain_indices[:, 0]) and
    #         #         y2 <= tf.reduce_max(brain_indices[:, 1]) and
    #         #         z2 <= tf.reduce_max(brain_indices[:, 2])
    #         #     )
    #         #     label = tf.cond(condition, lambda: tf.constant(1), lambda: tf.constant(0))
    #         fragment_batch.append(fragment)
    #         label_batch.append(label)
    #     yield K.constant(tf.stack(fragment_batch)), K.constant(tf.stack(label_batch))
    # for i, pos in enumerate(positions):
    #     x1, y1, z1, x2, y2, z2 = pos
    #     fragment = generated_img[0, x1:x2, y1:y2, z1:z2, :]
    #     yield fragment[None,...], labels[i][None,...]

# Create a combined generator
def combined_generator_12Net(brain_maps, model_feta,model_shapes,labels_to_image_model,batch_size):
    for label_map in generator_brain_label_maps(brain_maps):
        for fragment, label in generator_fragments_and_labels(label_map, model_feta,model_shapes,labels_to_image_model,param_3d.img_size_12 ,batch_size):
            yield fragment, label


import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops


def center_weighting_mask(shape, center_weight, boundary_weight):
    z, y, x = np.indices(shape)
    center = np.array([s // 2 for s in shape])
    dist = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
    max_dist = np.sqrt(center[0]**2 + center[1]**2 + center[2]**2)
    return np.clip(center_weight - (center_weight - boundary_weight) * (dist / max_dist), boundary_weight, center_weight)

def min_max_normalize(mask):
    min_val, max_val = np.min(mask), np.max(mask)
    return (mask - min_val) / (max_val - min_val) if max_val > min_val else mask

def consensus_based_combination(pred_list, weights, center_weight=1.0, boundary_weight=0.1):
    weight_mask = center_weighting_mask(pred_list[0].shape, center_weight, boundary_weight)
    normalized_masks = [min_max_normalize(mask) for mask in pred_list]
    combined = sum(mask * weight * weight_mask for mask, weight in zip(normalized_masks, weights))
    consensus_mask = np.sum(np.array(normalized_masks) > 0, axis=0) >= 2
    return np.where(consensus_mask, combined, 0)

def process_predictions(pred_list, param_3d):
    num_masks = len(pred_list)
    weights = np.linspace(1, num_masks, num_masks) / num_masks
    combined_mask = consensus_based_combination(pred_list, weights, center_weight=1.0, boundary_weight=0.1)
    binary_mask = combined_mask > (np.max(combined_mask) / 2)
    if np.sum(binary_mask) == 0:
        binary_mask = pred_list[0] == np.max(pred_list[0])
    return ndimage.binary_opening(ndimage.binary_closing(find_largest_component(binary_mask), structure=np.ones((3, 3, 3))), structure=np.ones((3, 3, 3))).astype(np.uint8)



# def process_predictions(pred_list, param_3d):
#     """
#     Process the prediction masks by combining them, finding large components,
#     and applying closing and hole filling.

#     Parameters:
#     - pred_list: List of prediction masks to be combined.
#     - param_3d: Parameters object containing min_area, median_area, max_area, and img_size_48.

#     Returns:
#     - Processed prediction mask.
#     """
#     # pred_192 = pred_list[-1]> np.max(pred_list[-1])-2
#     # structure = np.ones((2, 2,2))  # Define the structuring element
    
    
#     pred_192 = combine_masks_sum(pred_list)
#     pred_192 = pred_192 > np.max(pred_192)/2
#     # pred_192 = get_final_combined_mask(pred_list)
#     # max_val = np.max(pred_192)
#     # # thresholds = [max_val / i for i in range(50, 40, -10)]
#     # # threshold_masks = [(pred_192 > threshold).astype(int) for threshold in thresholds]
#     # pred_192 = pred_192> 0#/5#combine_masks_majority_voting(threshold_masks)
    
#     # pred_192 = find_large_components(pred_192,min_area=param_3d.min_area, max_area=param_3d.max_area)
#     pred_192 = find_largest_component(pred_192)#
#     pred_192 = ndimage.binary_opening(pred_192, structure=np.ones((3, 3, 3))).astype(np.uint8)
#     # # Perform closing to fill small holes
#     pred_192 = ndimage.binary_closing(pred_192, structure=np.ones((3, 3, 3))).astype(np.uint8)

#     # labeled_mask, num_features = ndimage.label(pred_192)
#     # min_area = param_3d.median_area if num_features > 1 else param_3d.min_area
#     # nib.save(nib.Nifti1Image(pred_192.astype(np.float32), np.array(img.geom.vox2world)), f"{folder_path}/cascade.nii.gz")
#     return pred_192
    

# def process_predictions(pred_list, param_3d):
#     """
#     Process the prediction masks by combining them, finding large components,
#     and applying closing and hole filling.

#     Parameters:
#     - pred_list: List of prediction masks to be combined.
#     - param_3d: Parameters object containing min_area, median_area, max_area, and img_size_48.

#     Returns:
#     - Processed prediction mask.
#     """
#     structure = np.ones((1, 1, 1))  # Define the structuring element
    
#     # Combine masks using majority voting
    
#     pred_192 = combine_masks_sum(pred_list)
#     max_val = np.max(pred_192)
#     thresholds = [max_val / i for i in range(100, 10, -10)]
#     threshold_masks = [(pred_192 > threshold).astype(int) for threshold in thresholds]
    
#     # threshold = np.max(pred_192)/100
#     # pred2 = pred_192 > threshold
#     pred1 = combine_masks_majority_voting(pred_list)
#     pred_192 = combine_masks_majority_voting([pred1]+threshold_masks)

#     # pred_192 = remove_outside_outliers(pred_192, iterations=3)
#     #combine_masks_majority_voting(pred_list)
    
#     # Adjust min_area based on the number of connected components
#     labeled_mask, num_features = ndimage.label(pred_192)
#     min_area = param_3d.median_area if num_features > 1 else param_3d.min_area
    
#     # Find large components
#     pred_192 = find_large_components(pred_192, min_area=min_area, max_area=param_3d.max_area)
    
#     # Apply morphological closing to fill concave regions and close bindings
#     pred_192 = ndimage.binary_closing(pred_192, structure=structure).astype(int)
    
#     # Fill small and big holes
#     pred_192 = ndimage.binary_fill_holes(pred_192).astype(int)
#     return pred_192
    
# def process_predictions(pred_list, param_3d):
#     """
#     Process the prediction masks by combining them, finding large components,
#     and applying closing and hole filling.

#     Parameters:
#     - pred_list: List of prediction masks to be combined.
#     - param_3d: Parameters object containing min_area, median_area, max_area, and img_size_48.

#     Returns:
#     - Processed prediction mask.
#     """
#     structure = np.ones((10, 10, 10)) # Define the structuring element
    
#     # Combine masks using majority voting
#     pred_192 = combine_masks_majority_voting(pred_list)
    
#     # Adjust min_area based on the number of connected components
#     labeled_mask, num_features = ndimage.label(pred_192)
#     min_area = param_3d.median_area if num_features > 1 else param_3d.min_area
    
#     # Find large components
#     pred_192 = find_large_components(pred_192, min_area=min_area, max_area=param_3d.max_area)
    
#     # Apply morphological closing to fill concave regions and close bindings
#     pred_192 = ndimage.binary_closing(pred_192, structure=structure).astype(int)
    
#     # Fill small and big holes
#     pred_192 = ndimage.binary_fill_holes(pred_192).astype(int)
    
#     return pred_192
    

# Example usage:
# Assuming pred_list and param_3d are already defined
# processed_pred = process_predictions(pred_list, param_3d)


# def process_pred_list(pred_list):
#     # Normalize each mask and sum them
#     pred_192 = sum(pred.astype(float) / np.sum(pred) for pred in pred_list)
    
#     # Apply thresholding    
#     if count_connected_components(pred_192) >1 : 
#         pred_192 = sum(pred.astype(float) / np.sum(pred) for pred in pred_list)
#         pred_192 = pred_192 > np.max(pred_192) / 20
#     elif count_connected_components(pred_192)==0:
#         pred_192= find_largest_component(combine_masks_majority_voting(pred_list))
#         return pred_192
#     pred_192 = pred_192 > np.max(pred_192) / 50
#     # Remove outliers if single connected component
#     if count_connected_components(pred_192) == 1:
#         pred_192 = remove_outside_outliers(pred_192, iterations=1)
#         # pred_192 = compare_masks_and_clip(pred_192, pred_192)
#         pred_192 = remove_outside_outliers(pred_192, iterations=1)
#     return pred_192
    

def gen_fragments(input_volume, patch_size,y):
    fragments = []
    # labels = generate_labels((192,192,192), positions, indices, param_3d.img_size_24, 4, input_volume)
    positions, indices = generate_position_map((192,192,192), param_3d.img_size_12, 4)

    for i in range(len(positions)):
        x1, y1, z1, x2, y2, z2 = positions[i]
        fragment = input_volume[0, x1:x2, y1:y2, z1:z2, :]
        yield fragment
    
def calculate_iou(window, mask):
    xmin, ymin, zmin, xmax, ymax, zmax = window
    window_mask = np.zeros_like(mask)
    window_mask[xmin:xmax,  ymin:ymax, zmin:zmax] = 1

    # mask_non_zero = mask > 0
    intersection = np.logical_and(mask, window_mask).sum()
    box_volume = np.sum(window_mask)
    iou = intersection / box_volume if box_volume != 0 else 0
    return iou

def draw_shapes_easy(
    shape,
    label_min=1,
    label_max=10,
    fwhm_min=1,
    fwhm_max=20,
    dtype=None,
    seed=None,
    **kwargs,
):
    # Data types.
    type_f = tf.float32
    type_i = tf.int32
    if dtype is None:
        dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    dtype = tf.dtypes.as_dtype(dtype)

    # Images and transforms.
    out = ne.utils.augment.draw_perlin_full(
        shape=(*shape, 2),
        fwhm_min=fwhm_min,
        fwhm_max=fwhm_max,
        isotropic=False,
        batched=False,
        featured=True,
        seed=seed,
        dtype=type_f,
        reduce=tf.math.reduce_max,
    )
    out = ne.utils.minmax_norm(out)

    num_label = tf.random.uniform(shape=(), minval=label_min, maxval=label_max + 1, dtype=type_i)
    out *= tf.cast(num_label, type_f)
    out = tf.cast(out, type_i)

    # Random relabeling. For less rare marginal label values.
    def reassign(x, max_in, max_out):
        lut = tf.random.uniform(shape=[max_in + 1], maxval=max_out, dtype=type_i)
        return tf.gather(lut, indices=x)

    # Add labels to break concentricity.
    a = reassign(out[..., 0:1], max_in=num_label, max_out=num_label)
    b = reassign(out[..., 1:2], max_in=num_label, max_out=num_label)
    out = reassign(a + b, max_in=2 * num_label - 2, max_out=num_label)
    # out = out[None,...]
    return tf.cast(out, dtype) if out.dtype != dtype else out

def draw_shapes(
    shape,
    num_label=16,
    warp_min=1,
    warp_max=20,
    dtype=None,
    seed=None,
    image_fwhm_min=20,
    image_fwhm_max=40,
    warp_fwhm_min=40,
    warp_fwhm_max=80,
):
    # Data types.
    type_fp = tf.float16
    type_int = tf.int32
    if dtype is None:
        dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    dtype = tf.dtypes.as_dtype(dtype)
    
    # Randomization.
    rand = np.random.default_rng(seed)
    seed = lambda: rand.integers(np.iinfo(np.int32).max, dtype=np.int32)
    prop = lambda: dict(isotropic=False, batched=False, featured=True, seed=seed(), dtype=type_fp, reduce=tf.math.reduce_max)
    
    # Images and transforms.
    v = ne.utils.augment.draw_perlin_full(
        shape=(*shape, 1),
        fwhm_min=image_fwhm_min, fwhm_max=image_fwhm_max, **prop(),
    )
    
    t = ne.utils.augment.draw_perlin_full(
        shape=(*shape, len(shape)), noise_min=warp_min, noise_max=warp_max,
        fwhm_min=warp_fwhm_min, fwhm_max=warp_fwhm_max, **prop(),
    )
    
    # Application and background.
    v = ne.utils.minmax_norm(v)
    v = vxm.utils.transform(v, t, fill_value=0)
    v = tf.math.ceil(v * (num_label - 1))

    return tf.cast(v, dtype) if v.dtype != dtype else v

import scipy.spatial.transform as spt
def bresenham_3d(start, end):
    """
    A basic implementation of Bresenham's line algorithm in 3D.
    Computes the points on a 3D line between two points.
    """
    points = []

    start = tf.cast(start, tf.int32)
    end = tf.cast(end, tf.int32)

    dx = abs(end[0] - start[0])
    dy = abs(end[1] - start[1])
    dz = abs(end[2] - start[2])

    x, y, z = start[0], start[1], start[2]
    x_inc = 1 if end[0] > start[0] else -1
    y_inc = 1 if end[1] > start[1] else -1
    z_inc = 1 if end[2] > start[2] else -1

    dx2 = dx * 2
    dy2 = dy * 2
    dz2 = dz * 2

    if dx >= dy and dx >= dz:  # X dominant
        err1 = dy2 - dx
        err2 = dz2 - dx
        for _ in range(dx):
            points.append([x, y, z])
            if err1 > 0:
                y += y_inc
                err1 -= dx2
            if err2 > 0:
                z += z_inc
                err2 -= dx2
            err1 += dy2
            err2 += dz2
            x += x_inc
    elif dy >= dx and dy >= dz:  # Y dominant
        err1 = dx2 - dy
        err2 = dz2 - dy
        for _ in range(dy):
            points.append([x, y, z])
            if err1 > 0:
                x += x_inc
                err1 -= dy2
            if err2 > 0:
                z += z_inc
                err2 -= dy2
            err1 += dx2
            err2 += dz2
            y += y_inc
    else:  # Z dominant
        err1 = dx2 - dz
        err2 = dy2 - dz
        for _ in range(dz):
            points.append([x, y, z])
            if err1 > 0:
                x += x_inc
                err1 -= dz2
            if err2 > 0:
                y += y_inc
                err2 -= dz2
            err1 += dx2
            err2 += dy2
            z += z_inc

    points.append([x, y, z])
    return tf.convert_to_tensor(points, dtype=tf.int32)
import tensorflow as tf

import tensorflow as tf
def rotate_point(point, angles):
    """ Rotate a point around the origin using specified angles (in degrees). """
    angles = np.radians(angles)
    rx = spt.Rotation.from_euler('x', angles[0]).as_matrix()
    ry = spt.Rotation.from_euler('y', angles[1]).as_matrix()
    rz = spt.Rotation.from_euler('z', angles[2]).as_matrix()
    rotation_matrix = rz @ ry @ rx
    return np.dot(rotation_matrix, point)

def draw_sphere(volume, center, radius, label):
    """ Draws a sphere in the volume at a given center with a given radius and label. """
    x, y, z = tf.meshgrid(
        tf.range(volume.shape[0], dtype=tf.float32),
        tf.range(volume.shape[1], dtype=tf.float32),
        tf.range(volume.shape[2], dtype=tf.float32),
        indexing='ij'
    )
    center = tf.cast(center, tf.float32)
    radius = tf.cast(radius, tf.float32)
    distance = tf.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    mask = distance <= radius
    volume = tf.where(mask, tf.cast(label, volume.dtype), volume)
    return volume
    
def draw_bones_only(
    shape,
    num_labels=16,
    num_bones=50,
    bone_length_range=(10, 80),
    bone_radius_range=(2, 6),
    label_min=8,
    label_max=16,
    fwhm_min=32,
    fwhm_max=128,
    dtype=None,
    seed=None,
    max_repeats=10,  # Maximum number of repetitions for each bone
    shift_range=50,   # Maximum shift range to reduce overlap
    **kwargs,
):
    # Seed for reproducibility
    if seed is not None:
        tf.random.set_seed(seed)
    type_f = tf.float32
    type_i = tf.int32
    dtype = tf.dtypes.as_dtype(dtype or tf.keras.mixed_precision.global_policy().compute_dtype)

    # Generate Perlin noise with two channels and normalize
    out = ne.utils.augment.draw_perlin_full(
        shape=(*shape, 2),
        fwhm_min=fwhm_min,
        fwhm_max=fwhm_max,
        isotropic=False,
        batched=False,
        featured=True,
        seed=seed,
        dtype=type_f,
        reduce=tf.math.reduce_max,
    )
    out = ne.utils.minmax_norm(out)
    out = out[..., 0]  # Remove the extra channel if not needed

    num_label = tf.random.uniform(shape=(), minval=label_min, maxval=label_max + 1, dtype=type_i)
    out *= tf.cast(num_label, type_f)
    out = tf.cast(out, type_i)

    # Function to draw a 3D line (stick/bone) with a given radius
    def draw_bone(volume, start, end, radius, label):
        # Bresenham's line algorithm in 3D
        points = bresenham_3d(start, end)
        for point in points:
            volume = draw_sphere(volume, point, radius, label)
        return volume

    # Generate bones
    for _ in range(num_bones):
        start = tf.stack([
            tf.random.uniform(shape=(), minval=0, maxval=shape[0], dtype=type_i),
            tf.random.uniform(shape=(), minval=0, maxval=shape[1], dtype=type_i),
            tf.random.uniform(shape=(), minval=0, maxval=shape[2], dtype=type_i)
        ])
        length = tf.random.uniform(shape=(), minval=bone_length_range[0], maxval=bone_length_range[1], dtype=type_i)
        direction = tf.random.normal(shape=[3])
        direction = direction / tf.norm(direction)  # Normalize to create a unit vector
        end = start + tf.cast(direction * tf.cast(length, direction.dtype), type_i)
        end = tf.clip_by_value(end, clip_value_min=0, clip_value_max=tf.constant(shape) - 1)
        radius = tf.random.uniform(shape=(), minval=bone_radius_range[0], maxval=bone_radius_range[1], dtype=type_i)
        label = tf.random.uniform(shape=(), minval=1, maxval=num_labels + 1, dtype=type_i)  # Random label

        # Number of repetitions for this bone
        num_repeats = tf.random.uniform(shape=(), minval=0, maxval=max_repeats + 1, dtype=type_i)
        
        # Draw the original bone and its repetitions
        for _ in range(num_repeats):
            # Apply a random shift to avoid overlap
            shift = tf.random.uniform(shape=[3], minval=-shift_range, maxval=shift_range, dtype=type_i)
            start_shifted = tf.clip_by_value(start + shift, clip_value_min=0, clip_value_max=tf.constant(shape) - 1)
            end_shifted = tf.clip_by_value(end + shift, clip_value_min=0, clip_value_max=tf.constant(shape) - 1)
            
            out = draw_bone(out, start_shifted, end_shifted, radius, label)

    return tf.cast(out, dtype) if out.dtype != dtype else out
    
def dynamic_resize(image, target_width=192):   

    fov = np.multiply(image.shape, image.geom.voxsize)

    new_voxsize = fov / target_width

    new_voxsize = np.max(new_voxsize[:2])  # ignore slice thickness
    return new_voxsize



from datetime import datetime
import os
from tensorflow.keras.callbacks import Callback

class PeriodicWeightsSaver(Callback):
    def __init__(self, filepath, save_freq=200, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.save_freq = save_freq
        os.makedirs(self.filepath, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            weights_path = os.path.join(self.filepath, f"weights_{timestamp}.h5")
            self.model.save_weights(weights_path)
            print(f"Saved weights to {weights_path}")


# class PeriodicWeightsSaver(tf.keras.callbacks.Callback):
#     def __init__(self, filepath, save_freq=200, **kwargs):
#         super().__init__(**kwargs)
#         self.filepath = filepath
#         self.save_freq = save_freq

#     def on_epoch_end(self, epoch, logs=None):
#         # Save the weights every `save_freq` epochs
#         if (epoch + 1) % self.save_freq == 0:
#             weights_path = os.path.join(self.filepath, f"weights_epoch_{epoch + 1}.h5")
#             self.model.save_weights(weights_path)
#             print(f"Saved weights to {weights_path}")

from tensorflow.keras.callbacks import Callback

import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback # Assuming this is a Keras Callback
import glob
from datetime import datetime

import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback # Assuming this is a Keras Callback
import glob
from datetime import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator # Added import
import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback # Assuming this is a Keras Callback
import glob
from datetime import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator # Added import

class CustomTensorBoard(Callback):
    def __init__(self, base_log_dir, models_dir, **kwargs):
        # Remove arguments that are specific to tf.keras.callbacks.TensorBoard
        # and not supported by the base Callback class.
        kwargs.pop('histogram_freq', None)
        kwargs.pop('write_graph', None)
        kwargs.pop('write_images', None)
        kwargs.pop('write_steps_per_second', None)
        kwargs.pop('update_freq', None)
        kwargs.pop('profile_batch', None)
        kwargs.pop('embeddings_freq', None)
        kwargs.pop('embeddings_metadata', None)

        super(CustomTensorBoard, self).__init__(**kwargs)
        self.base_log_dir = base_log_dir
        self.models_dir = models_dir

        # Define the fixed log directory path for TensorBoard events.
        # This will allow logs to append to existing files in this specific directory.
        self.train_log_dir = os.path.join(base_log_dir, 'train')
        
        # Initialize the file writer and resume point. They will be set in on_train_begin.
        self.file_writer = None
        self.latest_epoch_resume_point = 0 

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        # Ensure the fixed log directory exists
        os.makedirs(self.train_log_dir, exist_ok=True)
        
        # Create a new TensorBoard file writer for this run's log directory.
        # This will automatically create a new event file with a timestamp suffix
        # each time this method is called, as long as the previous writer was closed.
        self.file_writer = tf.summary.create_file_writer(self.train_log_dir)

        # Logic to determine the starting global step for logging in TensorBoard.
        # This will now be based on the latest step found in existing TensorBoard event files.
        self.latest_epoch_resume_point = 0
        
        # Find all event files in the train log directory
        event_files = sorted(
            [os.path.join(self.train_log_dir, f) for f in os.listdir(self.train_log_dir) if f.startswith("events.out.tfevents")],
            key=os.path.getmtime # Sort by modification time to get the latest ones
        )

        if event_files:
            max_step_found = -1
            # Iterate through all event files to find the absolute maximum step for 'loss'
            for event_file in event_files:
                acc = EventAccumulator(event_file)
                try:
                    acc.Reload()
                    # Check for 'scalars' tag, as tf.summary.scalar writes scalar values
                    if 'loss' in acc.Tags().get('scalars', []):
                        loss_events = acc.Scalars('loss')
                        if loss_events:
                            current_file_max_step = max(e.step for e in loss_events)
                            max_step_found = max(max_step_found, current_file_max_step)
                except Exception as e:
                    print(f"Warning: Could not read event file '{event_file}': {e}")
            
            if max_step_found != -1:
                # Set the resume point to the maximum step found + 1, ensuring continuity
                self.latest_epoch_resume_point = max_step_found + 1
                print(f"Detected logging resume point from existing TensorBoard logs: step {self.latest_epoch_resume_point}")
            else:
                print("No 'loss' scalar events found in existing log files. Starting logging from step 0.")
        else:
            print("No existing TensorBoard log files found. Starting new training run from step 0.")

        # The checkpoint parsing logic is no longer used for determining the resume point for logging.
        # It's assumed the model itself will handle loading weights based on `models_dir`
        # and that the logging step should align with the actual last logged event.

    def on_epoch_end(self, epoch, logs=None):
        # Close the previous writer to finalize the current event file
        if self.file_writer:
            self.file_writer.close()
    
        # Recreate the writer in the SAME log directory
        # This forces TensorFlow to create a NEW .v2 event file with a new timestamp
        self.file_writer = tf.summary.create_file_writer(self.train_log_dir)
    
        with self.file_writer.as_default():
            # Maintain global step continuity
            global_step = self.latest_epoch_resume_point + epoch
            for metric_name, value in logs.items():
                tf.summary.scalar(metric_name, value, step=global_step)
    
        self.file_writer.flush()
        print(f"✅ Logged metrics for epoch {epoch} into new event file in {self.train_log_dir}")

        
    def on_train_end(self, logs=None):
        # Close the file writer when training ends
        if self.file_writer:
            self.file_writer.close()
            # Explicitly set to None to ensure a new writer is created on next on_train_begin
            self.file_writer = None 



# class CustomTensorBoard(Callback):
#     def __init__(self, base_log_dir, models_dir, **kwargs):
#         super(CustomTensorBoard, self).__init__()
#         self.base_log_dir = base_log_dir
#         self.train_log_dir = os.path.join(base_log_dir, 'train')
#         self.models_dir = models_dir
#         self.latest_epoch = 0
#         latest_weight = max(glob.glob(os.path.join(self.train_log_dir, 'events*.v2')), key=os.path.getctime, default=None)
#         if latest_weight is not None:
#             latest_epoch = int(latest_weight.split('.')[-2])
#             print("latest epoch is ",10+latest_epoch)
#             self.latest_epoch = 10+latest_epoch
    

#     def set_model(self, model):
#         self.model = model

#     def on_train_begin(self, logs=None):
#         # Ensure the log directory exists
#         os.makedirs(self.train_log_dir, exist_ok=True)
        
#         # Load weights from the latest checkpoint to continue training
#         latest_checkpoint = tf.train.latest_checkpoint(self.train_log_dir)
#         if latest_checkpoint:
#             print(f"Resuming training from checkpoint: {latest_checkpoint}")
#             # Extract the epoch number from the checkpoint file name
#             # self.latest_epoch = int(latest_checkpoint.split('-')[-1].split('.')[0])
#             print(f"Resuming from epoch {self.latest_epoch}")

#     def on_epoch_end(self, epoch, logs=None):
#         # Log metrics for TensorBoard
#         with tf.summary.create_file_writer(self.train_log_dir, filename_suffix=".v2").as_default():
#             for metric_name, value in logs.items():
#                 tf.summary.scalar(metric_name, value, step=self.latest_epoch+epoch)
        
#         print(f"Logged metrics for epoch {epoch}")
        
# class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
#     def __init__(self, base_log_dir, **kwargs):
#          super(CustomTensorBoard, self).__init__(**kwargs)
#          self.base_log_dir = base_log_dir

#     def on_epoch_begin(self, epoch, logs=None):
#         # if epoch % self.histogram_freq == 0:  # Check if it's the start of a new set of 50 epochs
#         self.log_dir = self.base_log_dir
#         super().set_model(self.model)
def generator_brain_with_eyes(label_maps):
    rand = np.random.default_rng()
    label_maps = np.asarray(label_maps)
    
    while True:
        fg = rand.choice(label_maps)
        fg = place_two_spheres(fg, min_radius=3, max_radius=4, min_distance=15, max_distance=20, label=32)

        yield fg[None, ..., None]


import numpy as np
from scipy.ndimage import gaussian_filter

def attach_shifted_component_np(fg, min_shift=20, max_shift=50, stretch_factor=1.5, smoothing_sigma=2):
    shape = fg.shape
    # Calculate maximum possible shift in each dimension
    max_possible_shift = [s - min_shift for s in shape]
    
    # Randomly select shift values ensuring no overlap
    shift = np.random.randint(min_shift, max_shift + 1, size=3)
    
    # Ensure shift does not exceed available space
    shift = np.minimum(shift, max_possible_shift)
    
    # Create the shifted volume
    shifted_volume = np.roll(fg, shift=shift, axis=(0, 1, 2))
    
    # Find the center of the shifted volume
    center = np.array(shifted_volume.shape) // 2
    
    # Generate an ellipsoid mask
    z, y, x = np.indices(shifted_volume.shape)
    ellipsoid_mask = ((z - center[0]) ** 2 / (shape[0] // 4) ** 2 +
                      (y - center[1]) ** 2 / (shape[1] // 4) ** 2 +
                      (x - center[2]) ** 2 / (shape[2] // (4 * stretch_factor)) ** 2) <= 1
    
    # Smooth the ellipsoid mask to create softer boundaries
    ellipsoid_mask = gaussian_filter(ellipsoid_mask.astype(float), sigma=smoothing_sigma)
    
    # Threshold the smoothed ellipsoid to create the new component
    new_component = np.where(ellipsoid_mask > 0.5, 8, 0)
    
    # Combine original and new component, preserving the original non-zero values
    combined_volume = np.where(new_component > 0, new_component, fg)
    
    return combined_volume


def dilate_label_map(label_map, structure=None):
    """
    Dilates all non-zero elements in a 3D label map.
    
    Parameters:
    - label_map (np.ndarray): 3D array representing the label map with integer values.
    - structure (np.ndarray, optional): A structuring element for dilation. If None, a 3x3x3 cube is used.

    Returns:
    - dilated_label_map (np.ndarray): Dilated 3D label map.
    """
    # Create a binary mask where non-zero elements are marked as True
    binary_mask = label_map > 0
    
    # Define default structuring element if not provided (3x3x3 cube)
    if structure is None:
        structure = np.ones((10, 10, 10), dtype=bool)
    
    # Apply binary dilation
    dilated_mask = binary_dilation(binary_mask, structure=structure)
    
    # Create the new label map with dilated regions
    dilated_label_map = label_map.copy()
    
    # Replace the dilated positions with the original labels
    dilated_label_map[dilated_mask] = label_map[dilated_mask]
    
    return dilated_label_map
    
def extend_label_map_with_surfa(label_map, scale_factor=80, label_to_scale=15, img_size=[192, 192, 192]):
    label_data = np.array(label_map.data)
    label_area = (label_data == label_to_scale)
    label_data[label_data==15] = 0
    structure = np.ones((1, scale_factor, 1))
    label_area = binary_dilation(label_area, structure=structure)>0
    label_area = 15* label_area
    label_data = (label_area==0)*label_data + label_area
    print(np.unique(label_area),np.unique(label_data))
    return label_data


def generator_two_brain(label_maps):
    rand = np.random.default_rng()
    label_maps = np.asarray(label_maps)
    
    while True:
        fg = rand.choice(label_maps)        
        # Attach shifted component
        fg = attach_shifted_component_np(fg)
        
        yield fg[None, ..., None]
        
def generator_brain(label_maps):
    rand = np.random.default_rng()
    label_maps = np.asarray(label_maps)
    
    while True:
        fg = rand.choice(label_maps)
        yield fg[None, ..., None]

def get_brain(a):
    a_copy = np.copy(a)
    for i in range(len(a)):
        a_copy[i][a[i] >7 ] = 0
    return a_copy
    
def generator_brain_gmm(label_maps,cube_size=128):
    rand = np.random.default_rng()
    label_maps = np.asarray(label_maps)
    
    while True:
        fg = rand.choice(label_maps)
        fg = extract_centered_cube(fg,cube_size=cube_size)
        yield fg[None, ..., None]


def extract_random_cube(input_volume, cube_size=32):    
    x_max, y_max, z_max = input_volume.shape
    x1 = np.random.randint(0, x_max - cube_size + 1)
    y1 = np.random.randint(0, y_max - cube_size + 1)
    z1 = np.random.randint(0, z_max - cube_size + 1)
    
    x2 = x1 + cube_size
    y2 = y1 + cube_size
    z2 = z1 + cube_size
    cube = input_volume[x1:x2, y1:y2, z1:z2]
    return cube

def find_center_of_labels(input_volume, labels):
    coords = np.argwhere(np.isin(input_volume, labels))
    return np.mean(coords, axis=0).astype(int)

def extract_centered_cube(input_volume, cube_size=32):
    labels = [1, 2, 3, 4, 5, 6, 7]
    center_coords = find_center_of_labels(input_volume, labels)
    
    x_max, y_max, z_max = input_volume.shape
    half_size = cube_size // 2
    
    x1 = max(0, center_coords[0] - half_size)
    y1 = max(0, center_coords[1] - half_size)
    z1 = max(0, center_coords[2] - half_size)
    
    x2 = min(x_max, x1 + cube_size)
    y2 = min(y_max, y1 + cube_size)
    z2 = min(z_max, z1 + cube_size)
    
    cube = np.zeros((cube_size, cube_size, cube_size))
    cube[:x2-x1, :y2-y1, :z2-z1] = input_volume[x1:x2, y1:y2, z1:z2]
    
    return cube


def linear_soft_dice(a, b, foreground_weight=0.3, background_weight=0.7):
    # Calculate Soft Dice for foreground
    foreground_dice = soft_dice(a * b, b)
    
    # Calculate Soft Dice for background
    background_dice = soft_dice((1 - a) * (1 - b), 1 - b)
    
    # Linear combination of foreground and background Soft Dice coefficients
    total_dice = foreground_weight * foreground_dice + background_weight * background_dice
    
    return total_dice
    
class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='loss', value=1e-4, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if current < self.value:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: early stopping triggered. {self.monitor} reached {current}")
            self.model.stop_training = True

def softargmax(x, beta=1e10):
  x = tf.convert_to_tensor(x)
  x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
  return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)

def dynamic_resize(image, target_width=192):   

    fov = np.multiply(image.shape, image.geom.voxsize)

    new_voxsize = fov / target_width

    new_voxsize = np.max(new_voxsize[:2])  # ignore slice thickness
    return new_voxsize

def load_val(validation_folder_path,input_id):

    latest_images=[]
    latest_masks=[]
    b2_images=[]
    b3_images=[]
    b2_masks=[]
    b3_masks=[]
    mom_ids = []
    subfolders = [f.name for f in os.scandir(validation_folder_path) if f.is_dir()]
    
    for folder in subfolders:
        mom_str = folder.split("_")[1]
        if mom_str.isdigit():
            mom = int(mom_str)
        else:
            mom=0
        if input_id !=mom:
            continue
        folder_path = os.path.join(validation_folder_path, folder)
        # filename = os.path.join(folder_path,"image.nii.gz")
        filename = os.path.join(folder_path, "image.nii.gz") if os.path.exists(os.path.join(folder_path, "image.nii.gz")) else os.path.join(folder_path, "image.mgz")

        mask_filename = os.path.join(folder_path,"manual.nii.gz")
        image = sf.load_volume(filename)
        
        new_voxsize = [dynamic_resize(image)]*3
            
        orig_voxsize = image.geom.voxsize
        crop_img = image.resize([orig_voxsize[0],orig_voxsize[1],1], method="linear")
        crop_img = crop_img.resize(new_voxsize, method="linear").reshape([192, 192, 192])
        crop_data = crop_img.data
        
        mask = sf.load_volume(mask_filename).resize([orig_voxsize[0],orig_voxsize[1],1], method="linear")
        mask = mask.resize(new_voxsize).reshape([192, 192, 192, 1])
        mask.data[mask.data != 0] = 1

        
        # if abs(w-initial_epoch)<step_size:
        if mom<100:
            b2_images.append(crop_img)
            b2_masks.append(mask)
        else:
            b3_images.append(crop_img)
            b3_masks.append(mask)
        latest_images.append(crop_img)
        latest_masks.append(mask)
        mom_ids.append(mom)
            

    return mom_ids, latest_images, latest_masks , b2_images, b3_images, b2_masks, b3_masks
    
def load_random_val(validation_folder_path, count = 10):

    latest_images=[]
    latest_masks=[]
    b2_images=[]
    b3_images=[]
    b2_masks=[]
    b3_masks=[]
    mom_ids = []
    subfolders = [f.name for f in os.scandir(validation_folder_path) if f.is_dir()]
    i = 0 
    for folder in subfolders:
        i +=1

        if "rest" in folder or i > count:
            continue
            
        mom_str = folder.split("_")[1]
        if mom_str.isdigit():
            mom = int(mom_str)
        else:
            mom=0 
        folder_path = os.path.join(validation_folder_path, folder)
        filename = os.path.join(folder_path, "image.nii.gz") if os.path.exists(os.path.join(folder_path, "image.nii.gz")) else os.path.join(folder_path, "image.mgz")

        image = sf.load_volume(filename).reshape([192,192,192,1])
        new_voxsize = [dynamic_resize(image)]*3
            
        orig_voxsize = image.geom.voxsize
        crop_img = image.resize([orig_voxsize[0],orig_voxsize[1],1], method="linear")
        crop_img = crop_img.resize(new_voxsize, method="linear").reshape([192, 192, 192])
        crop_data = crop_img.data
        
        latest_images.append(crop_img)
        mom_ids.append(mom)
            

    return mom_ids, latest_images, latest_masks

def find_largest_component(mask):
    labeled_mask, num_features = ndi.label(mask)
    largest_component = None
    max_area = 0

    for region in regionprops(labeled_mask):
        if region.area > max_area:
            max_area = region.area
            largest_component = (labeled_mask == region.label)

    return largest_component if largest_component is not None else np.zeros_like(mask)

def find_large_components(mask, min_area=10000, max_area=60000):
    labeled_mask, num_features = ndi.label(mask)
    large_components = np.zeros_like(mask)

    for region in regionprops(labeled_mask):
        if min_area <= region.area <= max_area:
            large_components[labeled_mask == region.label] = 1

    return large_components

def count_connected_components(mask):
    """
    Counts the number of connected components in a binary mask.

    Parameters:
    mask (ndarray): Binary mask (2D array) where the connected components are to be counted.

    Returns:
    int: Number of connected components in the mask.
    """
    labeled_mask, num_features = label(mask, return_num=True)
    return num_features
    
def find_smallest_component(mask):
    labeled_mask, num_features = ndi.label(mask)
    smallest_component = None
    min_area = float('inf')  # Start with infinity to find the minimum

    for region in regionprops(labeled_mask):
        if region.area < min_area:
            min_area = region.area
            smallest_component = (labeled_mask == region.label)

    return smallest_component if smallest_component is not None else np.zeros_like(mask)

def fit_ellipsoid_to_largest_component(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        # Return default centroid as (32, 32, 32) and radii as 32 if no components are found
        return np.array([32, 32, 32]), np.array([32, 32, 32]), np.eye(3)
    
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid
    u, s, vh = svd(centered_coords, full_matrices=False)
    radii = np.max(np.abs(centered_coords @ vh.T), axis=0)
    # print(centroid, radii, vh)
    return centroid, radii, vh

def calculate_std_of_weighted_mask(weighted_mask):
    """
    Calculate the standard deviation of values in a weighted 3D mask.

    Args:
        weighted_mask (np.ndarray): 3D weighted mask.

    Returns:
        float: Standard deviation of the values in the weighted mask.
    """
    flattened_mask = weighted_mask.flatten()
    
    # Filter out zero values
    non_zero_values = flattened_mask[flattened_mask != 0]
    
    # Calculate the standard deviation of the non-zero values
    std_dev = np.std(non_zero_values)
    
    return std_dev


def apply_ellipsoid_filter(mask, centroid, radii, vh, scale_factor=1.0):
    filtered_mask = np.zeros_like(mask)
    coords = np.argwhere(mask)
    
    scaled_radii = radii * scale_factor
    inv_radii = 1 / scaled_radii
    inv_radii_matrix = np.diag(inv_radii)
    
    for coord in coords:
        transformed_coord = (coord - centroid) @ vh.T @ inv_radii_matrix
        if np.sum(transformed_coord**2) <= 1.0:
            filtered_mask[tuple(coord)] = 1
    
    return filtered_mask

def iterative_ellipsoid_filter(mask, min_fraction=0.96, step=0.05, max_iterations=100):
    largest_component = find_largest_component(mask)
    centroid, radii, vh = fit_ellipsoid_to_largest_component(largest_component)
    
    scale_factor = 1.0
    previous_mask = largest_component
    current_mask = apply_ellipsoid_filter(largest_component, centroid, radii, vh, scale_factor)
    
    iterations = 0
    while np.sum(current_mask) >= min_fraction * np.sum(previous_mask) and iterations < max_iterations:
        previous_mask = current_mask
        scale_factor -= step
        current_mask = apply_ellipsoid_filter(largest_component, centroid, radii, vh, scale_factor)
        iterations += 1
    
    return previous_mask


from scipy.ndimage import binary_dilation, binary_erosion
def morphological_cleaning(mask, iterations=3):
    """
    Apply morphological cleaning to remove outliers.

    Parameters:
    - mask: Input binary 3D mask.
    - iterations: Number of times to apply dilation and erosion.

    Returns:
    - Cleaned binary 3D mask.
    """
    structure = np.ones((3, 3, 3))  # Define the structure for morphological operations
    cleaned_mask = mask.copy()

    for _ in range(iterations):
        cleaned_mask = binary_dilation(cleaned_mask, structure=structure)
        cleaned_mask = iterative_ellipsoid_filter(cleaned_mask,min_fraction=0.95, step=0.04)
        cleaned_mask = binary_erosion(cleaned_mask, structure=structure)
    
    return cleaned_mask

def remove_outside_outliers(mask, iterations=2):
    # Dilate the mask to capture neighboring voxels
    
    eroded_mask = morphological_cleaning(mask,iterations=iterations)
    eroded_mask = find_large_components(eroded_mask,min_area=1000, max_area=60000)
    # Mask out everything outside the original brain mask
    cleaned_mask = mask.copy()
    cleaned_mask[eroded_mask == 0] = 0
    dilated_mask = binary_dilation(cleaned_mask, iterations=iterations)
    
    return cleaned_mask

def compare_masks_and_clip(first_mask, second_mask):
    # Create a mask of voxels that are within the boundaries of the first mask
    valid_mask = (second_mask == 1) & (first_mask == 1)
    
    # Clip the second mask to only keep voxels within the boundaries of the first mask
    clipped_mask = second_mask.copy()
    clipped_mask[~valid_mask] = 0
    
    return clipped_mask
    
def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2.0 * intersection) / (union) 

def my_hard_dice(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    dice = dice_coefficient(y_true_flat, y_pred_flat)
    return dice
    
def calculate_hard_dice(image, model, mask):
    prediction_one_hot = model.predict(image.data[None,...,None], verbose=0)
    predictions_argmax = np.argmax(prediction_one_hot, axis=-1)
    prediction = np.squeeze(predictions_argmax, axis=0)
    mask.data[mask.data != 0] = 1
    prediction = ndimage.binary_fill_holes(prediction).astype(int)
    return my_hard_dice(prediction.flatten(),mask.data.flatten())


def is_centered_3d(prediction, margins=(8, 8, 8)):
    try:
        # Sum projections along each axis
        summed_projection_axis0 = np.sum(prediction, axis=(1, 2))
        summed_projection_axis1 = np.sum(prediction, axis=(0, 2))
        summed_projection_axis2 = np.sum(prediction, axis=(0, 1))
        
        # Get non-zero coordinates for each axis
        non_zero_coords_axis0 = np.argwhere(summed_projection_axis0)
        non_zero_coords_axis1 = np.argwhere(summed_projection_axis1)
        non_zero_coords_axis2 = np.argwhere(summed_projection_axis2)
        
        # If any axis has no non-zero elements, return False
        if not non_zero_coords_axis0.size or not non_zero_coords_axis1.size or not non_zero_coords_axis2.size:
            return False
        
        # Get min and max coordinates for each axis
        min_coords_axis0 = non_zero_coords_axis0.min()
        max_coords_axis0 = non_zero_coords_axis0.max()
        
        min_coords_axis1 = non_zero_coords_axis1.min()
        max_coords_axis1 = non_zero_coords_axis1.max()
        
        min_coords_axis2 = non_zero_coords_axis2.min()
        max_coords_axis2 = non_zero_coords_axis2.max()
        
        # Check if non-zero elements have margins in all axes
        axis0_centered = (min_coords_axis0 >= margins[0]) and ((prediction.shape[0] - max_coords_axis0 - 1) >= margins[0])
        axis1_centered = (min_coords_axis1 >= margins[1]) and ((prediction.shape[1] - max_coords_axis1 - 1) >= margins[1])
        axis2_centered = (min_coords_axis2 >= margins[2]) and ((prediction.shape[2] - max_coords_axis2 - 1) >= margins[2])
        
        return axis0_centered and axis1_centered and axis2_centered
    except Exception as e:
        return False

def closest_margin_to_non_zero(mask):
    # Get coordinates of non-zero elements
    non_zero_coords = np.argwhere(mask)
    
    if non_zero_coords.size == 0:
        # No non-zero elements, return infinity as distance
        return float('inf'), float('inf'), float('inf')
    
    # Shape of the mask
    mask_shape = np.array(mask.shape)
    
    # Calculate distances to boundaries
    distances = []
    for dim in range(3):  # Assuming 3D mask
        min_distance = min(mask_shape[dim] - np.max(non_zero_coords[:, dim]), np.min(non_zero_coords[:, dim]))
        distances.append(min_distance)
    
    # Return the minimum distances to each boundary
    return distances[0], distances[1], distances[2]

def check_non_zero_within_bounds(mask, valid_positions):
    x1, y1, z1, x2, y2, z2 = valid_positions
    # Get indices of non-zero elements within the specified bounds
    non_zero_coords = np.argwhere(mask[x1:x2, y1:y2, z1:z2])
    
    # Check if there are any non-zero elements
    if non_zero_coords.size > 0:
        return 1
    else:
        return 0
        
def get_final_combined_mask(masks):
    """
    Return the final combined mask from a list of 3D masks.

    Args:
    masks (list of np.ndarray): List of 3D binary masks (0 or 1).

    Returns:
    np.ndarray: Final combined 3D mask.
    """
    if not masks:
        raise ValueError("The list of masks is empty.")

    final_mask = masks[-1] >= np.max(masks[-1])/2
    return final_mask.astype(int)

def get_last_two_masks(masks):
    """
    Return the last two masks from a list of 3D masks.

    Args:
    masks (list of np.ndarray): List of 3D binary masks (0 or 1).

    Returns:
    tuple of np.ndarray: The last two 3D masks.
    """
    if len(masks) < 2:
        raise ValueError("The list of masks must contain at least two masks.")
    
    return [masks[-2], masks[-1]]
    
def get_first_combined_mask(masks):
    """
    Return the final combined mask from a list of 3D masks.

    Args:
    masks (list of np.ndarray): List of 3D binary masks (0 or 1).

    Returns:
    np.ndarray: Final combined 3D mask.
    """
    if not masks:
        raise ValueError("The list of masks is empty.")

    final_mask = masks[0] > np.max(masks[0])/2
    return final_mask.astype(int)

def combine_masks_weighted_threshold(masks, method='max'):
    """
    Combine multiple 3D weighted masks and threshold based on the maximum or median value.
    
    Args:
    masks (list of np.ndarray): List of 3D weighted masks.
    method (str): Method to determine the threshold ('max' or 'median').
    
    Returns:
    np.ndarray: Combined and thresholded 3D mask.
    """
    # Sum the weighted masks
    combined_mask = np.sum(masks, axis=0)
    
    # Determine the threshold
    if method == 'max':
        threshold = np.max(combined_mask) / 2
    elif method == 'median':
        threshold = np.median(combined_mask)
    else:
        raise ValueError("Invalid method. Use 'max' or 'median'.")
    
    # Apply the threshold
    final_mask = combined_mask > threshold
    
    return final_mask.astype(int)

import numpy as np
from scipy.optimize import least_squares


def fit_sphere_to_nonzero_elements(mask, max_radius=16):
    """
    Fit spheres to the non-zero elements in a 3D mask for each connected component 
    and return a new mask containing these spheres.
    
    Args:
    mask (np.ndarray): 3D binary mask (0 or 1).
    max_radius (float): Maximum allowed radius for the sphere.
    
    Returns:
    np.ndarray: New mask with the fitted spheres.
    """
    from scipy.optimize import least_squares
    from scipy.ndimage import label

    # Label connected components in the mask
    labeled_mask, num_features = label(mask)
    
    new_mask = np.zeros_like(mask, dtype=int)
    
    for component in range(1, num_features + 1):
        # Get coordinates of the current component
        component_coords = np.column_stack(np.nonzero(labeled_mask == component))
        
        if component_coords.size == 0:
            continue
        
        # Function to compute the residuals (distance from sphere surface)
        def residuals(params, x, y, z):
            xc, yc, zc, r = params
            return np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2) - r
        
        # Initial guess for the sphere parameters
        x0, y0, z0 = np.mean(component_coords, axis=0)
        r0 = np.mean(np.linalg.norm(component_coords - np.array([x0, y0, z0]), axis=1))
        r0 = min(r0, max_radius)  # Ensure initial guess radius is within the max radius
        initial_guess = [x0, y0, z0, r0]
        
        # Perform least squares optimization with constraint on radius
        result = least_squares(residuals, initial_guess, args=(component_coords[:, 0], component_coords[:, 1], component_coords[:, 2]), bounds=([-np.inf, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, max_radius]))
        
        # Extract the fitted sphere parameters
        xc, yc, zc, r = result.x
        
        # Define a grid of coordinates
        xx, yy, zz = np.indices(mask.shape)
        
        # Calculate the distance from the center for each point in the grid
        distances = np.sqrt((xx - xc)**2 + (yy - yc)**2 + (zz - zc)**2)
        
        # Set points within the radius to 1
        new_mask[distances <= r] = 1
    
    return new_mask
    


def create_ellipsoid_around(brain_volume, margin=2, stretch_factors=(1.0, 5.0, 1.5)):
    """
    Create an ellipsoid around the brain volume.

    Args:
        brain_volume (tf.Tensor): 3D tensor of the brain volume.
        margin (int): Additional margin to add around the brain bounding box.
        stretch_factors (tuple): Factors to stretch the ellipsoid along each axis (z, y, x).

    Returns:
        tf.Tensor: 3D tensor of the ellipsoid mask.
    """
    # Find the bounding box of the brain
    non_zero_indices = tf.where(brain_volume > 0)
    min_z, max_z = tf.reduce_min(non_zero_indices[:, 0]), tf.reduce_max(non_zero_indices[:, 0])
    min_y, max_y = tf.reduce_min(non_zero_indices[:, 1]), tf.reduce_max(non_zero_indices[:, 1])
    min_x, max_x = tf.reduce_min(non_zero_indices[:, 2]), tf.reduce_max(non_zero_indices[:, 2])

    # Calculate the center and radii of the ellipsoid
    center = tf.cast([(max_z + min_z) / 2, (max_y + min_y) / 2, (max_x + min_x) / 2], tf.float32)
    radii = tf.cast([(max_z - min_z) / 2, (max_y - min_y) / 2, (max_x - min_x) / 2], tf.float32) + margin

    # Apply the stretch factors to the radii
    radii = radii * tf.constant(stretch_factors, dtype=tf.float32)

    # Create a grid of coordinates
    z = tf.range(brain_volume.shape[0], dtype=tf.float32)
    y = tf.range(brain_volume.shape[1], dtype=tf.float32)
    x = tf.range(brain_volume.shape[2], dtype=tf.float32)
    
    z_grid, y_grid, x_grid = tf.meshgrid(z, y, x, indexing='ij')

    # Calculate the ellipsoid formula
    ellipsoid = ((x_grid - center[2]) ** 2 / radii[2] ** 2 +
                 (y_grid - center[1]) ** 2 / radii[1] ** 2 +
                 (z_grid - center[0]) ** 2 / radii[0] ** 2) <= 1

    return tf.cast(ellipsoid, tf.uint8)
    
    
def combine_masks_weighted_threshold_quartile(masks):
    """
    Combine multiple 3D weighted masks and threshold based on the fourth quartile (75th percentile) of non-zero weights.
    
    Args:
    masks (list of np.ndarray): List of 3D weighted masks.
    
    Returns:
    np.ndarray: Combined and thresholded 3D mask.
    """
    # Combine masks and flatten to work with non-zero values
    combined_mask = np.sum(masks, axis=0).flatten()
    
    # Remove zero values
    combined_mask_nonzero = combined_mask[combined_mask > 0]
    
    # Determine the threshold as the 75th percentile of non-zero values
    threshold = np.percentile(combined_mask_nonzero, 75)
    
    # Apply the threshold to create the final mask
    final_mask = np.sum(masks, axis=0) > threshold
    
    return final_mask.astype(int)

def combine_masks_median_threshold_median(masks):
    """
    Combine multiple 3D weighted masks and threshold based on the median of non-zero weights.
    
    Args:
    masks (list of np.ndarray): List of 3D weighted masks.
    
    Returns:
    np.ndarray: Combined and thresholded 3D mask.
    """
    # Combine masks and flatten to work with non-zero values
    combined_mask = np.sum(masks, axis=0).flatten()
    
    # Remove zero values
    combined_mask_nonzero = combined_mask[combined_mask > 0]
    
    # Determine the threshold as the median of non-zero values
    threshold = np.median(combined_mask_nonzero)
    
    # Apply the threshold to create the final mask
    final_mask = np.sum(masks, axis=0) > threshold
    
    return final_mask.astype(int)


def combine_masks_max_threshold_median(masks):
    """
    Combine multiple 3D weighted masks and threshold based on the median of non-zero weights.
    
    Args:
    masks (list of np.ndarray): List of 3D weighted masks.
    
    Returns:
    np.ndarray: Combined and thresholded 3D mask.
    """
    # Combine masks and flatten to work with non-zero values
    combined_mask = np.sum(masks, axis=0).flatten()
    
    # Determine the threshold as the median of non-zero values
    threshold = np.max(combined_mask)
    
    # Apply the threshold to create the final mask
    final_mask = np.sum(masks, axis=0) >= threshold
    
    return final_mask.astype(int)
    
def combine_masks_weighted_average_descending(masks, weights):
    """
    Combine multiple 3D masks using weighted averaging with descending weights.
    
    Args:
    masks (list of np.ndarray): List of 3D probability masks.
    weights (list of float): List of weights for each mask.
    
    Returns:
    np.ndarray: Combined 3D mask.
    """
    descending_weights = sorted(weights, reverse=True)
    weighted_masks = [mask * weight for mask, weight in zip(masks, descending_weights)]
    combined_mask = np.sum(weighted_masks, axis=0) / np.sum(descending_weights)
    return (combined_mask > 0.5).astype(int)


def combine_masks_weighted_average_ascending(masks, weights):
    """
    Combine multiple 3D masks using weighted averaging with ascending weights.
    
    Args:
    masks (list of np.ndarray): List of 3D probability masks.
    weights (list of float): List of weights for each mask.
    
    Returns:
    np.ndarray: Combined 3D mask.
    """
    ascending_weights = sorted(weights)
    weighted_masks = [mask * weight for mask, weight in zip(masks, ascending_weights)]
    combined_mask = np.sum(weighted_masks, axis=0) / np.sum(ascending_weights)
    return (combined_mask > 0.5).astype(int)


def is_centered_in_plane(prediction, margins=(8, 8, 8, 8)):
    summed_projection_axis2 = np.sum(prediction, axis=2)
    
    summed_projection_axis0 = np.sum(summed_projection_axis2, axis=0)
    summed_projection_axis1 = np.sum(summed_projection_axis2, axis=1)
    
    non_zero_coords_axis0 = np.argwhere(summed_projection_axis0)
    non_zero_coords_axis1 = np.argwhere(summed_projection_axis1)
    
    min_coords_axis0 = np.min(non_zero_coords_axis0, axis=0)
    max_coords_axis0 = np.max(non_zero_coords_axis0, axis=0)
    
    min_coords_axis1 = np.min(non_zero_coords_axis1, axis=0)
    max_coords_axis1 = np.max(non_zero_coords_axis1, axis=0)
    
    axis0_centered = np.all(min_coords_axis0 >= margins[0]) and np.all(summed_projection_axis0.shape - max_coords_axis0 - 1 >= margins[1])
    axis1_centered = np.all(min_coords_axis1 >= margins[2]) and np.all(summed_projection_axis1.shape - max_coords_axis1 - 1 >= margins[3])
    
    return axis0_centered and axis1_centered


def combine_masks_intersection(masks):
    """
    Combine multiple 3D masks using intersection.
    
    Args:
    masks (list of np.ndarray): List of 3D binary masks (0 or 1).
    
    Returns:
    np.ndarray: Combined 3D mask.
    """
    combined_mask = np.logical_and.reduce(masks)
    return combined_mask.astype(int)

def combine_masks_union(masks):
    """
    Combine multiple 3D masks using union.
    
    Args:
    masks (list of np.ndarray): List of 3D binary masks (0 or 1).
    
    Returns:
    np.ndarray: Combined 3D mask.
    """
    combined_mask = np.logical_or.reduce(masks)
    return combined_mask.astype(int)
    
def combine_masks_sum(masks):
    """
    Combine multiple 3D masks by summing their values.
    
    Args:
    masks (list of np.ndarray): List of 3D binary masks (0 or 1).
    
    Returns:
    np.ndarray: Combined 3D mask with summed values.
    """
    combined_mask = np.sum(masks, axis=0)
    return combined_mask

def combine_masks_top_75_percent(masks):
    """
    Combine multiple 3D masks and threshold to only allow the top 25% of unique values.
    
    Args:
    masks (list of np.ndarray): List of 3D masks.
    
    Returns:
    np.ndarray: Combined and thresholded 3D mask.
    """
    # Sum the masks
    combined_mask = np.sum(masks, axis=0)
    
    # Get unique values and sort them
    unique_values = np.unique(combined_mask)
    
    # Determine the threshold for the top 25% of unique values
    threshold_index = int(0.25 * len(unique_values))
    threshold_value = unique_values[threshold_index]
    # Create the final mask based on the threshold
    
    final_mask = combined_mask >= threshold_value
    # print(np.sum(final_mask))
    return final_mask.astype(int)
    
import numpy as np

from scipy.ndimage import label, find_objects

def remove_small_components(mask, min_size=1000):
    """
    Remove connected components smaller than the specified size from a binary mask.
    Ensures that no component of size >= min_size is removed.

    Parameters:
    - mask: 3D binary numpy array representing the mask.
    - min_size: Minimum size of components to keep (e.g., 10000 voxels).

    Returns:
    - filtered_mask: 3D binary numpy array with small components removed.
    """
    # Label connected components
    labeled_mask, num_features = label(mask)
    
    # Get the size of each connected component
    component_sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
    
    # Find components smaller than min_size
    small_components = [i + 1 for i, size in enumerate(component_sizes) if size < min_size]
    
    # Create a mask for keeping large components
    large_components_mask = np.isin(labeled_mask, small_components, invert=True).astype(int)
    
    return large_components_mask


def combine_masks_all_occurrences(pred_list):
    """
    Combine binary masks such that only voxels appearing in all masks are set to 1.

    Parameters:
    - pred_list: List of 3D binary numpy arrays representing the masks.

    Returns:
    - combined_mask: 3D binary numpy array where voxels that appear in all masks
                      are set to 1, others are set to 0.
    """
    # Convert list of masks to a 4D numpy array (new axis for masks)
    stacked_masks = np.stack(pred_list, axis=-1)
    
    # Find the number of masks
    num_masks = len(pred_list)
    
    # Create combined mask where voxel is 1 if it appears in all masks
    combined_mask = np.all(stacked_masks == 1, axis=-1).astype(int)
    
    return combined_mask


def combine_masks_majority_voting(masks):
    """
    Combine multiple 3D masks using majority voting.
    
    Args:
    masks (list of np.ndarray): List of 3D binary masks (0 or 1).
    
    Returns:
    np.ndarray: Combined 3D mask.
    """
    # Set any mask with a sum less than 1000 to all 1s
    processed_masks = []
    for mask in masks:
        if np.sum(mask) < 1000:
            mask = np.ones_like(mask)
        processed_masks.append(mask)

    # Stack the processed masks along a new axis
    masks_stack = np.stack(processed_masks, axis=-1)
    
    # Apply majority voting
    combined_mask = np.sum(masks_stack, axis=-1) > (len(masks) / 2)
    
    return combined_mask.astype(int)

    
def calculate_weights(masks):
    """
    Calculate the weights for each mask based on the number of connected components.

    Args:
    masks (list of np.ndarray): List of 3D binary masks (0 or 1).

    Returns:
    np.ndarray: Weights for each mask.
    """
    weights = []
    for mask in masks:
        weight = 1.0
        for i in range(mask.shape[0]):
            if count_connected_components(mask[i]) > 1:
                weight *= 0.5
        weights.append(weight)
    return np.array(weights)

def combine_masks_weighted_majority_voting(masks):
    """
    Combine multiple 3D masks using weighted majority voting.

    Args:
    masks (list of np.ndarray): List of 3D binary masks (0 or 1).

    Returns:
    np.ndarray: Combined 3D mask.
    """
    weights = calculate_weights(masks)
    masks_stack = np.stack(masks, axis=-1)
    
    # Apply weights to masks
    weighted_masks_stack = masks_stack * weights.reshape(1, 1, 1, -1)
    
    # Sum along the new axis and apply majority voting
    combined_mask = np.sum(weighted_masks_stack, axis=-1) > (np.sum(weights) / 2)
    return combined_mask.astype(int)
    
    
def first_stage_prediction(img, mask, mom, positions_48):
    min_size , max_size = get_min_max_size(1,mom)
    detection = False
    combined_model_48 = get_model(models[3]) 

    for min_size in tqdm(range(min_size, 15000, -4000)):
        detection, valid_position_index_192, cube_48, mask_48, first_pred_192 = find_brain_48(positions_48, min_size, max_size, combined_model_48, img, mask)
        if detection:
            break

    return detection, valid_position_index_192, cube_48, mask_48, first_pred_192

def second_stage_prediction(img, mask, cube_48, mask_48, valid_position_index_192, mom, positions_24):
    detection = False
    combined_model_24 = get_model(models[2]) 
    min_size , max_size = get_min_max_size(2,mom)
    for min_size in tqdm(range(min_size, 1000, -4000)):
        detection, pred_48, valid_position_index_48, cube_24, mask_24, second_pred_192 = find_brain_24(positions_24, min_size, max_size, combined_model_24, 
                                                                                                       cube_48, mask_48, valid_position_index_192, img, mask)
        if detection:
            break
    if not detection:
        raise ValueError("No mask found in this stage!")
    return detection, pred_48, valid_position_index_48, cube_24, mask_24, second_pred_192

def third_stage_prediction(img, mask, cube_24, mask_24, cube_48, mask_48, valid_position_index_192, valid_position_index_48, mom, positions_12):
    detection = False
    combined_model_12 = get_model(models[1]) 
    min_size , max_size = get_min_max_size(3,mom)
    pred_24 = np.zeros_like(mask_24)
    print("min size",min_size)
    for min_size in tqdm(range(min_size, 3000, -2000)):
        detection, pred_24, valid_position_index_24, cube_12, mask_12, third_pred_192 = find_brain_12(positions_12, min_size, max_size, combined_model_12, 
                                                                                                     cube_24, mask_24, cube_48, mask_48, img, mask, valid_position_index_192, valid_position_index_48)
        if detection:
            break
    if not detection:
        raise ValueError("No mask found in this stage!")
    return detection, pred_24, valid_position_index_24, cube_12, mask_12, third_pred_192

def fourth_stage_prediction(img, mask, cube_12, mask_12, cube_24, mask_24, cube_48, mask_48, valid_position_index_24, valid_position_index_48, valid_position_index_192, mom, positions_6):
    detection = False
    combined_model_6 = get_model(models[0]) 
    min_size , max_size = get_min_max_size(4,mom)
    
    list_pred_192 = []

    for min_size in tqdm(range(min_size, 5000, -2000)):
        detection, pred_12, valid_position_index_12, cube_6, fourth_pred_192 = find_brain_6(positions_6, min_size, max_size, combined_model_6, 
                                                                                            cube_12, mask_12, cube_24, mask_24, cube_48, mask_48, img, mask, 
                                                                                            valid_position_index_24, valid_position_index_48, valid_position_index_192)
        if detection:
            break
            
    if not detection:
        raise ValueError("No mask found in this stage!")
    return detection, pred_12, valid_position_index_12, cube_6, fourth_pred_192


def compute_weights(length):
    """
    Compute weights for the masks, where earlier masks have higher weights.
    
    Args:
    length (int): Number of masks.
    
    Returns:
    list of float: Weights for each mask.
    """
    weights = np.linspace(1, 0.5, length)
    weights = weights / np.sum(weights)
    return weights.tolist()

def compute_logarithmic_weights(length):
    """
    Compute logarithmically decreasing weights for the masks, where earlier masks have higher weights.
    
    Args:
    length (int): Number of masks.
    
    Returns:
    list of float: Logarithmically decreasing weights for each mask.
    """
    if length <= 0:
        raise ValueError("Length must be a positive integer.")
    
    weights = np.logspace(0, -1, length)
    weights = weights / np.sum(weights)
    return weights.tolist()



def zero_out_margins(image, margin_size=20):
    """
    Sets the margins of a 3D image to zero.
    
    Args:
    image (np.ndarray): Input 3D image.
    margin_size (int): Size of the margin to zero out.
    
    Returns:
    np.ndarray: Image with zeroed-out margins.
    """
    # Make a copy of the image to avoid modifying the original
    new_image = np.copy(image)
    
    # Set the margins to zero
    new_image[:margin_size, :, :] = 0
    new_image[-margin_size:, :, :] = 0
    new_image[:, :margin_size, :] = 0
    new_image[:, -margin_size:, :] = 0
    new_image[:, :, :margin_size] = 0
    new_image[:, :, -margin_size:] = 0
    
    return new_image


def zero_out_margins(image, margin_size=20):
    """
    Sets the margins of a 3D image to zero.
    
    Args:
    image (np.ndarray): Input 3D image.
    margin_size (int): Size of the margin to zero out.
    
    Returns:
    np.ndarray: Image with zeroed-out margins.
    """
    # Make a copy of the image to avoid modifying the original
    new_image = np.copy(image)
    
    # Set the margins to zero
    new_image[:margin_size, :, :] = 0
    new_image[-margin_size:, :, :] = 0
    new_image[:, :margin_size, :] = 0
    new_image[:, -margin_size:, :] = 0
    new_image[:, :, :margin_size] = 0
    new_image[:, :, -margin_size:] = 0
    
    return new_image

def randomly_brighten_3d_image(image, max_brightness_level=0.2):
    """
    Randomly increase the brightness of a 3D image.

    Args:
        image (tf.Tensor): 3D image tensor.
        max_brightness_level (float): Maximum level to increase the brightness.

    Returns:
        tf.Tensor: Brightened 3D image.
    """
    adjust = tf.random.uniform([]) > 0.5
    
    if adjust:
        # Randomly choose to brighten or darken (50% chance for each)
        brighten = tf.random.uniform([]) > 0.5

        # Choose a random brightness adjustment level between 0 and max_brightness_level
        brightness_level = tf.random.uniform([], minval=0, maxval=max_brightness_level)

        if brighten:
            # Increase the brightness
            adjusted_image = image * (1 + brightness_level)
        else:
            # Decrease the brightness
            adjusted_image = image * (1 - brightness_level)

        # Ensure the adjusted image is within the original value range
        adjusted_image = tf.clip_by_value(adjusted_image, tf.reduce_min(image), tf.reduce_max(image))
    else:
        # No change
        adjusted_image = image

    return adjusted_image
    

def add_noise_to_3d_image(image,  maxval=0.5, noise_type='gaussian'):
    """
    Add noise to a 3D image using TensorFlow.

    Args:
        image (tf.Tensor): 3D image tensor.
        noise_type (str): Type of noise to add ('gaussian' or 'salt_and_pepper').

    Returns:
        tf.Tensor: Noisy 3D image.
    """
    if noise_type not in ['gaussian', 'salt_and_pepper']:
        raise ValueError("Invalid noise type. Supported types are 'gaussian' and 'salt_and_pepper'.")
    
    # Choose a random noise level between 0 and the maximum value in the image
    noise_level = tf.random.uniform([], minval=0, maxval=maxval)

    noisy_image = tf.identity(image)

    if noise_type == 'gaussian':
        mean = 0.0
        std = noise_level * tf.math.reduce_std(image)
        gaussian_noise = tf.random.normal(shape=image.shape, mean=mean, stddev=std, dtype=image.dtype)
        noisy_image += gaussian_noise
    
    elif noise_type == 'salt_and_pepper':
        prob = noise_level
        random_values = tf.random.uniform(shape=image.shape)
        salt_pepper_noise = tf.where(random_values < prob / 2, tf.zeros_like(image), 
                                     tf.where(random_values < prob, tf.ones_like(image) * tf.reduce_max(image), image))
        noisy_image = salt_pepper_noise

    # Ensure the noisy image is within the original value range
    noisy_image = tf.clip_by_value(noisy_image, tf.reduce_min(image), tf.reduce_max(image))
    
    return noisy_image

def add_bias_to_3d_image(image, max_num_patches = 35, max_patch_size = 80, bias_type='patch', bias_level=0.3):
    """
    Add bias to a 3D image using TensorFlow.

    Args:
        image (tf.Tensor): 5D image tensor with shape (batch_size, depth, height, width, channels).
        bias_type (str): Type of bias to add ('uniform', 'gradient', 'sinusoidal', 'patch').
        bias_level (float): Level of the bias for non-random types.

    Returns:
        tf.Tensor: 5D image with added bias.
    """
    if bias_type not in ['uniform', 'gradient', 'sinusoidal', 'patch']:
        raise ValueError("Invalid bias type. Supported types are 'uniform', 'gradient', 'sinusoidal', and 'patch'.")
    
    # Remove batch and channel dimensions for bias calculation
    input_shape = tf.shape(image)
    image_3d = tf.squeeze(image, axis=[0, -1])  # Shape (depth, height, width)

    # Choose a random bias level between 0 and the maximum value in the image
    random_bias_level = tf.random.uniform([], minval=0, maxval=tf.reduce_max(image_3d) * bias_level)

    biased_image_3d = tf.identity(image_3d)

    if bias_type == 'uniform':
        uniform_bias = random_bias_level
        biased_image += uniform_bias
    
    elif bias_type == 'gradient':
        gradient_bias = tf.linspace(0.0, random_bias_level, num=image.shape[0])
        gradient_bias = tf.reshape(gradient_bias, [image.shape[0], 1, 1])
        biased_image += gradient_bias

    elif bias_type == 'patch':
        # Define patch size and number of patches
        min_dim = tf.minimum(tf.minimum(input_shape[1], input_shape[2]), input_shape[3])
        max_patch_size = 80#tf.minimum(min_dim // 2, 64)
        patch_size = tf.random.uniform([], minval=4, maxval=max_patch_size, dtype=tf.int32)
        num_patches = tf.random.uniform([], minval=1, maxval=max_num_patches, dtype=tf.int32)

        for _ in range(num_patches):
            # Randomly choose patch location ensuring it fits within the image bounds
            z_start = tf.random.uniform([], minval=0, maxval=input_shape[1] - patch_size, dtype=tf.int32)
            y_start = tf.random.uniform([], minval=0, maxval=input_shape[2] - patch_size, dtype=tf.int32)
            x_start = tf.random.uniform([], minval=0, maxval=input_shape[3] - patch_size, dtype=tf.int32)
            
            # Create Gaussian bias patch
            zz, yy, xx = tf.meshgrid(
                tf.range(tf.cast(patch_size, tf.float32), dtype=tf.float32),
                tf.range(tf.cast(patch_size, tf.float32), dtype=tf.float32),
                tf.range(tf.cast(patch_size, tf.float32), dtype=tf.float32),
                indexing='ij'
            )
            zz -= tf.cast(patch_size, tf.float32) / 2
            yy -= tf.cast(patch_size, tf.float32) / 2
            xx -= tf.cast(patch_size, tf.float32) / 2
            sigma = tf.cast(patch_size, tf.float32) / 4.0
            gaussian_bias_patch = random_bias_level * tf.exp(-(tf.square(zz) + tf.square(yy) + tf.square(xx)) / (2.0 * tf.square(sigma)))
            
            # Add Gaussian bias patch to the image
            patch_indices = tf.stack(tf.meshgrid(
                tf.range(z_start, z_start + patch_size),
                tf.range(y_start, y_start + patch_size),
                tf.range(x_start, x_start + patch_size),
                indexing='ij'
            ), axis=-1)
            patch_indices = tf.reshape(patch_indices, [-1, 3])
            updates = tf.reshape(gaussian_bias_patch, [-1])
            biased_image_3d = tf.tensor_scatter_nd_sub(biased_image_3d, patch_indices, updates)

    # Ensure the biased image is within the original value range
    biased_image_3d = tf.clip_by_value(biased_image_3d, tf.reduce_min(image_3d), tf.reduce_max(image_3d))

    # Reshape back to the original 5D shape
    biased_image = tf.reshape(biased_image_3d, input_shape[1:4])
    biased_image = tf.expand_dims(tf.expand_dims(biased_image, axis=0), axis=-1)
    
    return biased_image
    



def create_and_shift_ellipsoid(fg, min_margin=-2, max_margin=2, min_shift=24, max_shift=32):
    """
    Create an ellipsoid around the brain with a random margin and shift it randomly in any direction.

    Parameters:
    fg (tf.Tensor): The foreground 3D tensor representing the brain.
    min_margin (int): Minimum margin value.
    max_margin (int): Maximum margin value.
    min_shift (int): Minimum shift value.
    max_shift (int): Maximum shift value.

    Returns:
    tf.Tensor: The created and shifted ellipsoid.
    """
    # Randomly choose a margin
    margin = tf.random.uniform([], min_margin, max_margin, dtype=tf.int32)

    # Find the bounding box of the brain
    non_zero_indices = tf.where(fg > 0)
    min_z, max_z = tf.reduce_min(non_zero_indices[:, 0]), tf.reduce_max(non_zero_indices[:, 0])
    min_y, max_y = tf.reduce_min(non_zero_indices[:, 1]), tf.reduce_max(non_zero_indices[:, 1])
    min_x, max_x = tf.reduce_min(non_zero_indices[:, 2]), tf.reduce_max(non_zero_indices[:, 2])

    # Calculate the center and radii of the ellipsoid
    center = tf.cast([(max_z + min_z) / 2, (max_y + min_y) / 2, (max_x + min_x) / 2], tf.float32)
    radii = tf.cast([(max_z - min_z) / 2, (max_y - min_y) / 2, (max_x - min_x) / 2], tf.float32) + tf.cast(margin, tf.float32)

    # Create a grid of coordinates
    z = tf.range(fg.shape[0], dtype=tf.float32)
    y = tf.range(fg.shape[1], dtype=tf.float32)
    x = tf.range(fg.shape[2], dtype=tf.float32)
    
    z_grid, y_grid, x_grid = tf.meshgrid(z, y, x, indexing='ij')

    # Calculate the ellipsoid formula
    ellipsoid = ((x_grid - center[2]) ** 2 / radii[2] ** 2 +
                 (y_grid - center[1]) ** 2 / radii[1] ** 2 +
                 (z_grid - center[0]) ** 2 / radii[0] ** 2) <= 1

    elips = tf.cast(ellipsoid, tf.uint8)
    
    # Randomly choose a shift value and direction
    shift_value = tf.random.uniform([], min_shift, max_shift, dtype=tf.int32)
    axis = np.random.choice([0, 1, 2])

    # Shift the ellipsoid
    shifted_elips = tf.roll(elips, shift=shift_value, axis=axis)

    return tf.cast(shifted_elips, tf.int32)
    
def longest_non_zero_spread(image):
    non_zero_indices = np.argwhere(image != 0)
    return 0 if non_zero_indices.size == 0 else (non_zero_indices.max(axis=0) - non_zero_indices.min(axis=0) + 1).max()

def shortest_non_zero_spread(image):
    nz = np.argwhere(image != 0)
    return 0 if nz.size == 0 else (nz.max(0) - nz.min(0) + 1).min()

def SSIMMeasure(y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    
def SSIMLoss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)  # Example SSIM calculation
    return -(ssim - 2.0)

import tensorflow as tf
from tensorflow.keras import backend as K

def MSELoss(y_true, y_pred):
    y_true_shape = K.shape(y_true)
    y_pred_shape = K.shape(y_pred)
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return -1+mse
import numpy as np
from scipy.spatial.transform import Rotation as R

def create_sphere_mask(shape, center, radius):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    z_center, y_center, x_center = center
    sphere_mask = (x - x_center)**2 + (y - y_center)**2 + (z - z_center)**2 <= radius**2
    return sphere_mask

def find_nonzero_voxels(volume):
    return np.argwhere(volume > 0)

def calculate_bounding_box(non_zero_voxels):
    min_coords = np.min(non_zero_voxels, axis=0)
    max_coords = np.max(non_zero_voxels, axis=0)
    return min_coords, max_coords

def random_rotation():
    r = R.random()
    return r.as_matrix()

def rotate_point(point, rotation_matrix):
    return np.dot(rotation_matrix, point)

def place_two_spheres(volume, min_radius=2, max_radius=5, min_distance=8, max_distance=12, label=8):
    non_zero_voxels = find_nonzero_voxels(volume)
    
    if len(non_zero_voxels) < 2:
        raise ValueError("Not enough non-zero elements to place spheres.")
    
    min_coords, max_coords = calculate_bounding_box(non_zero_voxels)

    # Calculate the initial possible positions for the spheres
    mid_x = (min_coords[2] + max_coords[2]) // 2
    mid_y = (min_coords[1] + max_coords[1]) // 2
    mid_z = (min_coords[0] + max_coords[0]) // 2
    
    # Randomly generate distance between spheres
    distance = np.random.randint(min_distance, max_distance + 1)
    
    # Randomly decide the side for the spheres
    direction = np.random.choice([-1, 1])
    
    center1 = np.array([mid_z, mid_y, mid_x - direction * distance / 2])
    center2 = np.array([mid_z, mid_y, mid_x + direction * distance / 2])

    # Randomly generate radii for the spheres
    radius1 = np.random.randint(min_radius, max_radius + 1)
    radius2 = np.random.randint(min_radius, max_radius + 1)

    # Apply random rotation
    rotation_matrix = random_rotation()
    center1_rotated = rotate_point(center1 - np.array([mid_z, mid_y, mid_x]), rotation_matrix) + np.array([mid_z, mid_y, mid_x])
    center2_rotated = rotate_point(center2 - np.array([mid_z, mid_y, mid_x]), rotation_matrix) + np.array([mid_z, mid_y, mid_x])

    # Ensure the centers are within the bounds of the volume
    center1_rotated = np.clip(center1_rotated, radius1, np.array(volume.shape) - radius1 - 1)
    center2_rotated = np.clip(center2_rotated, radius2, np.array(volume.shape) - radius2 - 1)

    # Place the spheres with random radii
    sphere_mask1 = create_sphere_mask(volume.shape, center1_rotated, radius1)
    sphere_mask2 = create_sphere_mask(volume.shape, center2_rotated, radius2)
    
    volume[sphere_mask1] = label
    volume[sphere_mask2] = label
    
    return volume

def randomly_darken_3d_image(image, max_darkness_level=0.9):
    """
    Randomly decrease the brightness of a 3D image.

    Args:
        image (tf.Tensor): 3D image tensor.
        max_darkness_level (float): Maximum level to decrease the brightness.

    Returns:
        tf.Tensor: Darkened 3D image.
    """
    # Ensure the image values are between 0 and 1
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    # Print the min and max values of the original image for debugging

    # Choose a random darkness level between 0 and max_darkness_level
    darkness_level = tf.random.uniform([], minval=0, maxval=max_darkness_level)

    # Decrease the brightness of the image
    darkened_image = image * (1 - darkness_level)

    # Print the min and max values of the darkened image for debugging

    return darkened_image


def attach_shifted_component(fg, shift_range=(5, 20)):
    fg = fg[0,...,0]
    fg = tf.cast(fg, tf.int32)
    
    # Randomly select shift values
    shift = tf.random.uniform(shape=(3,), minval=shift_range[0], maxval=shift_range[1], dtype=tf.int32)
    
    # Create the shifted volume
    shifted_volume = tf.roll(fg, shift=shift, axis=[0, 1, 2])
    
    # Label new component as 8
    new_component = tf.where(shifted_volume > 0, 8, 0)
    
    # Combine original and new component
    combined_volume = tf.where(new_component > 0, new_component, fg)
    
    return combined_volume[None,...,None]

import tensorflow as tf
import matplotlib.pyplot as plt
def create_sphere(center, radius, shape):
    z, y, x = tf.meshgrid(tf.range(shape[2], dtype=tf.float32),
                          tf.range(shape[1], dtype=tf.float32),
                          tf.range(shape[0], dtype=tf.float32),
                          indexing='ij')
    distance = tf.sqrt(tf.square(x - center[0]) + tf.square(y - center[1]) + tf.square(z - center[2]))
    sphere_mask = distance <= radius
    return sphere_mask

def create_ellipsoid(center, radii, shape):
    z, y, x = tf.meshgrid(tf.range(shape[2], dtype=tf.float32),
                          tf.range(shape[1], dtype=tf.float32),
                          tf.range(shape[0], dtype=tf.float32),
                          indexing='ij')
    ellipsoid_mask = (tf.square(x - center[0]) / tf.square(radii[0]) +
                      tf.square(y - center[1]) / tf.square(radii[1]) +
                      tf.square(z - center[2]) / tf.square(radii[2])) <= 1
    return ellipsoid_mask

def add_random_shape(volume, volume_shape, label_range):
    center = tf.random.uniform(shape=(3,), minval=0, maxval=volume_shape[0], dtype=tf.int32).numpy()
    shape_type = random.choice(['sphere', 'ellipsoid'])
    if shape_type == 'sphere':
        radius = random.randint(20, 50)
        mask = create_sphere(center, float(radius), volume_shape)
    elif shape_type == 'ellipsoid':
        radii = [float(random.randint(20, 50)) for _ in range(3)]
        mask = create_ellipsoid(center, radii, volume_shape)
    else:
        raise ValueError("Unsupported shape type")

    label = random.randint(1, label_range)
    volume = tf.where(mask, label, volume)
    return volume

def add_noise_and_smooth(volume, sigma=2):
    volume_np = volume.numpy()  # Convert TensorFlow tensor to NumPy array
    volume_np = gaussian_filter(volume_np, sigma=sigma)  # Apply Gaussian filter
    volume_np = np.round(volume_np).astype(np.int32)  # Round and convert to int32
    return tf.convert_to_tensor(volume_np, dtype=tf.int32)  # Convert back to TensorFlow tensor

def draw_layer_elipses(shape=(192, 192, 192), num_labels=16, num_shapes=100, sigma=2):
    volume = tf.zeros(shape, dtype=tf.int32)
    for _ in range(num_shapes):
        volume = add_random_shape(volume, shape, num_labels)
    
    volume = add_noise_and_smooth(volume, sigma=sigma)
    return volume

def show_slice(volume, slice_idx):
    plt.imshow(volume[:, :, slice_idx].numpy(), cmap='jet')
    plt.colorbar()
    plt.title(f'Slice {slice_idx}')
    plt.show()


def closest_distance_to_boundary(volume, valid_position_index_192, prev_shape):
    x1, y1, z1, x2, y2, z2 = valid_position_index_192
    volume = extract_cube(volume, x1, y1, z1, x2, y2, z2, cube_size=prev_shape)
    non_zero_indices = np.argwhere(volume > 0)
    
    if non_zero_indices.size == 0:
        return float('inf')

    # print("aaa")
    # ne.plot.volume3D(volume, slice_nos=ms)
    # print("bbb")
    min_coords = non_zero_indices.min(axis=0)
    max_coords = non_zero_indices.max(axis=0)
    shape = np.array(volume.shape)
    
    return np.min(np.concatenate((min_coords, shape - max_coords - 1)))

    
def zoom_and_pad_binary_mask(mask, zoom_factor=0.8):
    """
    Zoom in on a binary mask and pad it back to its original shape.

    Parameters:
    - mask: 3D binary numpy array representing the mask.
    - zoom_factor: Float representing the zoom level (e.g., 2.0 for 2x zoom).

    Returns:
    - padded_mask: 3D binary numpy array zoomed and padded back to original shape.
    """
    original_shape = mask.shape  # Store the original shape of the mask

    # Zoom in on the mask using nearest-neighbor interpolation
    zoomed_mask = scipy.ndimage.zoom(mask.astype(float), zoom_factor, order=0)
    zoomed_mask = np.round(zoomed_mask).astype(int)  # Round to ensure binary values

    # Calculate the cropping size based on the zoomed mask
    zoomed_shape = zoomed_mask.shape
    crop_size = [min(zs, os) for zs, os in zip(zoomed_shape, original_shape)]  # Ensure crop fits in the original mask

    # Crop the zoomed mask to the original shape or smaller
    start_zoomed = [(zs - cs) // 2 for zs, cs in zip(zoomed_shape, crop_size)]
    end_zoomed = [sz + cs for sz, cs in zip(start_zoomed, crop_size)]
    cropped_zoomed_mask = zoomed_mask[start_zoomed[0]:end_zoomed[0], 
                                      start_zoomed[1]:end_zoomed[1], 
                                      start_zoomed[2]:end_zoomed[2]]

    # Create a zero-filled array of the original shape
    padded_mask = np.zeros(original_shape, dtype=int)

    # Calculate where to place the cropped zoomed mask in the padded mask (centered)
    start_pad = [(os - cs) // 2 for os, cs in zip(original_shape, crop_size)]
    end_pad = [sp + cs for sp, cs in zip(start_pad, crop_size)]

    # Place the cropped zoomed mask into the padded mask
    padded_mask[start_pad[0]:end_pad[0], start_pad[1]:end_pad[1], start_pad[2]:end_pad[2]] = cropped_zoomed_mask
    
    return padded_mask


def crop_and_pad_to_original(volume, zoom_factor=0.8):
    """
    Crop the center of the 3D volume and pad it back to its original shape.

    Parameters:
    - volume: 3D numpy array representing the medical image.
    - crop_size: Tuple indicating the size of the region to crop from the center.

    Returns:
    - padded_volume: 3D numpy array of the volume cropped and padded to original shape.
    """
    original_shape = volume.shape  # Store the original shape of the volume

    # Zoom in on the volume
    zoomed_volume = scipy.ndimage.zoom(volume, zoom_factor)

    # Calculate the cropping size based on the zoomed volume
    zoomed_shape = zoomed_volume.shape
    crop_size = [min(zs, os) for zs, os in zip(zoomed_shape, original_shape)]  # Ensure crop fits in the original volume

    # Crop the zoomed volume to the original shape or smaller
    start_zoomed = [(zs - cs) // 2 for zs, cs in zip(zoomed_shape, crop_size)]
    end_zoomed = [sz + cs for sz, cs in zip(start_zoomed, crop_size)]
    cropped_zoomed_volume = zoomed_volume[start_zoomed[0]:end_zoomed[0], 
                                          start_zoomed[1]:end_zoomed[1], 
                                          start_zoomed[2]:end_zoomed[2]]

    # Create a zero-filled array of the original shape
    padded_volume = np.zeros(original_shape)

    # Calculate where to place the cropped zoomed volume in the padded volume (centered)
    start_pad = [(os - cs) // 2 for os, cs in zip(original_shape, crop_size)]
    end_pad = [sp + cs for sp, cs in zip(start_pad, crop_size)]

    # Place the cropped zoomed volume into the padded volume
    padded_volume[start_pad[0]:end_pad[0], start_pad[1]:end_pad[1], start_pad[2]:end_pad[2]] = cropped_zoomed_volume
    
    return padded_volume
    
def consensus_based_combination(pred_list, weights, center_weight=1.0, boundary_weight=0.1):
    def min_max_normalize(mask):
        if mask.dtype == bool:
            return mask.astype(float)  # Convert boolean to float without normalization
        min_val, max_val = np.min(mask), np.max(mask)
        return (mask - min_val) / (max_val - min_val) if max_val > min_val else mask
        
    
    weight_mask = center_weighting_mask(pred_list[0].shape, center_weight, boundary_weight)
    normalized_masks = [min_max_normalize(mask) for mask in pred_list]
    combined = sum(mask * weight * weight_mask for mask, weight in zip(normalized_masks, weights))
    consensus_mask = np.sum(np.array(normalized_masks) > 0, axis=0) >= 2
    return np.where(consensus_mask, combined, 0)




def refine_prediction1(crop_img, mask, model, model_128, folder, new_image_size=(192, 192, 192), margin=0, cube_size=128):
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
    folder_path = os.path.join("results", folder)
    os.makedirs(folder_path, exist_ok=True)
    nib.save(nib.Nifti1Image(crop_img, np.eye(4)), os.path.join(folder_path, 'image.nii.gz'))

    # Step 1: Initial Prediction
    # Binarize the mask
    mask.data[mask.data != 0] = 1
    nib.save(nib.Nifti1Image(mask.astype(np.int32), np.eye(4)), os.path.join(folder_path, 'mask.nii.gz'))

    # Compute mask center (using the provided find_bounding_box function)
    ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)
    print(crop_img.shape)
    
    # Make an initial prediction
    prediction_one_hot = model.predict(crop_img[None, ...], verbose=0)
    initial_prediction = np.argmax(prediction_one_hot, axis=-1)[0]
    ne.plot.volume3D(crop_img, slice_nos=ms)
    print("Initial Prediction Result:")

    labeled, num_components = ndimage.label(initial_prediction > 0)
    largest_mask = labeled == np.argmax(ndimage.sum(initial_prediction > 0, labeled, range(num_components + 1)))
    initial_prediction = ndi.binary_fill_holes(largest_mask)
    nib.save(nib.Nifti1Image(initial_prediction.astype(np.int32), np.eye(4)), os.path.join(folder_path, 'initial_prediction.nii.gz'))

    ne.plot.volume3D(initial_prediction, slice_nos=ms)
    print("first step: ",my_hard_dice(mask.data, initial_prediction))

    # Step 2: Use find_bounding_box function to get the bounding box
    x1, y1, z1, x2, y2, z2 = find_bounding_box(initial_prediction, cube_size=cube_size)
    cube = extract_cube(crop_img, x1, y1, z1, x2, y2, z2, cube_size=128)


    pred_192 = np.zeros((192,192,192))

    ms = np.mean(np.column_stack(np.nonzero(mask)), axis=0).astype(int)
    ne.plot.volume3D(cube, slice_nos=ms)

    # Step 3: Re-run the Model with the cropped image
    prediction_cropped_one_hot = model_128.predict(cube[None, ...], verbose=0)
    final_prediction = np.argmax(prediction_cropped_one_hot, axis=-1)[0]
    pred_192[x1:x2, y1:y2, z1:z2] = final_prediction
    pred_192[pred_192==1]=1
    
    labeled, num_components = ndimage.label(pred_192 > 0)
    largest_mask = labeled == np.argmax(ndimage.sum(pred_192 > 0, labeled, range(num_components + 1)))
    largest_mask = ndi.binary_fill_holes(largest_mask)
    pred_192 = largest_mask
    ne.plot.volume3D(pred_192, slice_nos=ms)
    print("second step: ",my_hard_dice(mask.data, pred_192))
    # Step 4: Resize the final prediction to the original crop_img size
    # final_prediction_resized = np.resize(final_prediction, (192, 192, 192))
    nib.save(nib.Nifti1Image(pred_192.astype(np.int32), np.eye(4)), os.path.join(folder_path, 'second_prediction.nii.gz'))
    return pred_192

