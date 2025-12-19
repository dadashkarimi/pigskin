import math
import itertools
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import einops as E
import sys
import torch
import os
import random
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import re
import torch.nn.functional as F
import surfa as sf
from medpy.metric.binary import dc, hd95 # Ensure these are imported for compute_stats

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- IMPORTANT: DEFINE YOUR MODEL'S EXPECTED SLICE SIZE HERE ---
# This is the HxW that universeg expects for its 2D inputs.
MODEL_INPUT_SLICE_SIZE = (192, 192) # (Height, Width)

# --- Define a common depth if your data has varying depths, or just infer it later ---
# Based on your error, 352 seems to be a common depth.
# We will use the *original* depth of each volume during processing.

from universeg import universeg
model = universeg(pretrained=True)
# If using DataParallel (for multiple GPUs), uncomment this:
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
#     model = torch.nn.DataParallel(model)
model = model.to(device)


def visualize_tensors(tensors, col_wrap=8, col_names=None, title=None):
    M = len(tensors)
    N = len(next(iter(tensors.values())))

    cols = col_wrap
    rows = math.ceil(N/cols) * M

    d = 2.5
    fig, axes = plt.subplots(rows, cols, figsize=(d*cols, d*rows))
    if rows == 1:
      axes = axes.reshape(1, cols)

    for g, (grp, tensors) in enumerate(tensors.items()):
        for k, tensor in enumerate(tensors):
            col = k % cols
            row = g + M*(k//cols)
            x = tensor.detach().cpu().numpy().squeeze()
            ax = axes[row,col]
            if len(x.shape) == 2:
                ax.imshow(x,vmin=0, vmax=1, cmap='gray')
            else:
                ax.imshow(E.rearrange(x,'C H W -> H W C'))
            if col == 0:
                ax.set_ylabel(grp, fontsize=16)
            if col_names is not None and row == 0:
                ax.set_title(col_names[col])

    for i in range(rows):
        for j in range(cols):
            ax = axes[i,j]
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

    if title:
        plt.suptitle(title, fontsize=20)

    plt.tight_layout()

def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred.long()
    y_true = y_true.long()
    score = 2*(y_pred*y_true).sum() / (y_pred.sum() + y_true.sum())
    return score.item()

@torch.no_grad()
def inference(model, image, label, support_images, support_labels):
    image, label = image.to(device), label.to(device)

    logits = model(
        image[None],
        support_images[None],
        support_labels[None]
    )[0]

    soft_pred = torch.sigmoid(logits)
    hard_pred = soft_pred.round().clip(0,1)

    score = dice_score(hard_pred, label)

    return {'Image': image,
            'Soft Prediction': soft_pred,
            'Prediction': hard_pred,
            'Ground Truth': label,
            'score': score}


# ---------- Dataset ----------
class PigSliceDataset(Dataset):
    def __init__(self, folders, axis=2):
        self.samples = []
        self.axis = axis

        for folder in folders:
            try:
                # Load and resample 3D volumes so their slices match MODEL_INPUT_SLICE_SIZE
                pig_anat_data = self.load_and_resample_volume_for_support(folder, 'anat', MODEL_INPUT_SLICE_SIZE)
                pig_mask_data = self.load_and_resample_volume_for_support(folder, 'anat_brain_olfactory_mask', MODEL_INPUT_SLICE_SIZE, is_mask=True)

                num_slices = pig_anat_data.shape[self.axis] # H, W, D or D, H, W depending on axis
                for i in range(num_slices):
                    if self.axis == 0:
                        img_slice = pig_anat_data[i, :, :]
                        mask_slice = pig_mask_data[i, :, :]
                    elif self.axis == 1:
                        img_slice = pig_anat_data[:, i, :]
                        mask_slice = pig_mask_data[:, i, :]
                    else: # self.axis == 2 (axial)
                        img_slice = pig_anat_data[:, :, i]
                        mask_slice = pig_mask_data[:, :, i]

                    if np.any(mask_slice > 0):
                        self.samples.append((img_slice, mask_slice))

            except FileNotFoundError as fnfe:
                print(f"⚠️ Skipping {folder}: {fnfe}")
            except Exception as e:
                print(f"⚠️ Skipping {folder} due to unexpected error: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, mask = self.samples[idx]

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask

    @staticmethod
    def load_and_resample_volume_for_support(folder_path, file_name, target_slice_hw, is_mask=False):
        # target_slice_hw is (Height, Width) for the 2D slices
        for ext in ['.nii.gz', '.nii']:
            file_path = os.path.join(folder_path, file_name + ext)
            if os.path.exists(file_path):
                vol_nii = nib.load(file_path)
                vol_data_orig = vol_nii.get_fdata(dtype=np.float32)

                original_h, original_w, original_d = vol_data_orig.shape

                # We need to resample the H and W dimensions to MODEL_INPUT_SLICE_SIZE
                # The depth (D) remains as the original depth of this specific volume.
                # The interpolation will happen on the (H, W, D) representation.
                vol_tensor = torch.tensor(vol_data_orig, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # 1, 1, H_orig, W_orig, D_orig

                # Target size for interpolation: (target_H, target_W, original_D)
                # Note: F.interpolate expects sizes in the order of the tensor dimensions
                # which are H, W, D after squeezing batch/channel.
                # So here, size refers to (D, H, W) for a 3D input to interpolate for a 2D slice
                # but our tensor is (H,W,D) and we interpolate (H,W)
                
                # Resample H and W to MODEL_INPUT_SLICE_SIZE, keeping original D
                resized_vol_tensor = F.interpolate(vol_tensor,
                                                   size=(target_slice_hw[0], target_slice_hw[1], original_d),
                                                   mode='nearest' if is_mask else 'trilinear',
                                                   align_corners=False if not is_mask else None)

                resized_data = resized_vol_tensor.squeeze().cpu().numpy()

                if is_mask:
                    resized_data = (resized_data > 0.5).astype(np.uint8)
                else:
                    resized_data = np.clip(resized_data, 0, 255)

                return resized_data # Returns the 3D volume, resampled in H and W
        raise FileNotFoundError(f"{file_name} not found in {folder_path}")


# ---------- Prediction Function (updated for consistent input size and output resampling) ----------
def predict_universeg_3d(subject_folder, support_images, support_labels, model, batch_size=16):
    anat_path = None
    for ext in ['.nii.gz', '.nii']:
        test_path = os.path.join(subject_folder, "image" + ext)
        if os.path.exists(test_path):
            anat_path = test_path
            break
    if anat_path is None:
        raise FileNotFoundError(f"No image.nii[.gz] found in {subject_folder}")

    vol_nii = nib.load(anat_path)
    vol_data_orig = vol_nii.get_fdata(dtype=np.float32)
    original_affine = vol_nii.affine # Keep original affine for saving
    original_header = vol_nii.header # Keep original header for saving if needed

    original_h, original_w, original_d = vol_data_orig.shape

    # 1. Prepare input volume for model: Resample H and W to MODEL_INPUT_SLICE_SIZE
    vol_tensor_for_model = torch.tensor(vol_data_orig, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # 1, 1, H_orig, W_orig, D_orig

    resized_h_for_model = MODEL_INPUT_SLICE_SIZE[0]
    resized_w_for_model = MODEL_INPUT_SLICE_SIZE[1]

    # Only interpolate if dimensions are different
    if original_h != resized_h_for_model or original_w != resized_w_for_model:
        vol_tensor_for_model = F.interpolate(vol_tensor_for_model,
                                             size=(resized_h_for_model, resized_w_for_model, original_d), # Keep original D
                                             mode='trilinear',
                                             align_corners=False)

    vol_data_for_model = vol_tensor_for_model.squeeze().cpu().numpy()
    vol_data_for_model = np.clip(vol_data_for_model, 0, 255) # Apply clipping

    # Permute to (D, 1, H, W) for slice iteration
    vol_tensor_slices = torch.tensor(vol_data_for_model, dtype=torch.float32).permute(2, 0, 1).unsqueeze(1).to(device) / 255.0

    num_slices = vol_tensor_slices.shape[0] # This is original_d

    # Initialize pred_vol with the H and W dimensions of the model's output slices
    # Its shape will be (D, 1, MODEL_INPUT_SLICE_SIZE[0], MODEL_INPUT_SLICE_SIZE[1])
    pred_vol_model_dims = torch.zeros_like(vol_tensor_slices, dtype=torch.uint8)

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, num_slices, batch_size), desc=f"Infer {os.path.basename(subject_folder)} (All Slices)"):
            batch_idxs = list(range(i, min(i + batch_size, num_slices)))
            batch_tensor = vol_tensor_slices[batch_idxs]

            B = batch_tensor.shape[0]

            sup_img = support_images.unsqueeze(0).expand(B, -1, -1, -1, -1)
            sup_lbl = support_labels.unsqueeze(0).expand(B, -1, -1, -1, -1)

            logits = model(batch_tensor, sup_img, sup_lbl)
            preds = torch.sigmoid(logits).round().clip(0, 1).byte()

            pred_vol_model_dims[batch_idxs] = preds

    # 2. Reconstruct 3D prediction from model-sized slices (H_model, W_model, D_orig)
    # Undo the permute to get (H_model, W_model, D_orig)
    pred_np_model_dims = pred_vol_model_dims.squeeze(1).permute(1, 2, 0).cpu().numpy()

    # 3. Resample predicted 3D volume back to ORIGINAL IMAGE DIMENSIONS
    # This is the crucial step to match the "main image" for evaluation/saving.
    pred_tensor_orig_dims = torch.tensor(pred_np_model_dims, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # 1, 1, H_model, W_model, D_orig

    # Target size for final interpolation: (original_h, original_w, original_d)
    final_pred_tensor = F.interpolate(pred_tensor_orig_dims,
                                      size=(original_h, original_w, original_d),
                                      mode='nearest', # Use nearest for masks
                                      align_corners=None)

    final_pred_np = final_pred_tensor.squeeze().cpu().numpy().astype(np.uint8)


    save_path = os.path.join(subject_folder, "universeg.nii.gz")
    # Use the original affine and header from the input image for the output NIfTI
    nib.save(nib.Nifti1Image(final_pred_np, original_affine, original_header), save_path)
    print(f"✅ Saved prediction (all slices, original dimensions) to {save_path}")
    torch.cuda.empty_cache()

# ---------- Evaluation ----------
def load_nifti(path): return nib.load(path).get_fdata()
def binarize(x): return (x > 0).astype(np.uint8)
def iou(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return intersection / union if union > 0 else 0

def compute_stats(gt, pred):
    gt_bin = binarize(gt)
    pred_bin = binarize(pred)
    if np.sum(gt_bin) == 0 and np.sum(pred_bin) == 0:
        dice_val = 1.0
        hd95_val = 0.0
    elif np.sum(gt_bin) == 0 or np.sum(pred_bin) == 0:
        dice_val = 0.0
        hd95_val = np.inf
    else:
        dice_val = dc(pred_bin, gt_bin)
        hd95_val = hd95(pred_bin, gt_bin)
    return {
        "Dice": dice_val,
        "Hausdorff95": hd95_val,
        "IoU": iou(pred_bin, gt_bin)
    }

def extract_numeric_id(folder_name):
    m = re.search(r'(\d+)', folder_name)
    return m.group(1) if m else None

# ---------- Support Setup ----------
folders_path = [
    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/template/",
    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/81-T2",
    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/75",
    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/79-T2",
    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/79",
    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/78",
    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/106_6month/",
    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/82",
    "/gpfs/fs001/cbica/home/dadashkj/upenn_pigAnatomical/93"
]

N_SUPPORT_TARGET = 700

random.seed(42)
random.shuffle(folders_path)
support_folders = folders_path
d_support = PigSliceDataset(support_folders)
print("d_support",d_support)
n_support = min(N_SUPPORT_TARGET, len(d_support))

if n_support == 0:
    raise ValueError(f"No support samples available from {len(support_folders)} folders with non-zero masks. Cannot proceed.")

support_indices = random.sample(range(len(d_support)), n_support) if n_support < len(d_support) else list(range(len(d_support)))

support_images, support_labels = zip(*[d_support[i] for i in support_indices])
support_images = torch.stack(support_images).to(device)
support_labels = torch.stack(support_labels).to(device)

print(f"Using {n_support} support samples (target was {N_SUPPORT_TARGET}).")


# ---------- Run Prediction ----------
base_dir_for_test_subjects = "results"

all_test_subject_folders = [
    os.path.join(base_dir_for_test_subjects, f)
    for f in os.listdir(base_dir_for_test_subjects)
    if os.path.isdir(os.path.join(base_dir_for_test_subjects, f))
]
random.shuffle(all_test_subject_folders)

train_ids = {extract_numeric_id(os.path.basename(f)) for f in support_folders if extract_numeric_id(os.path.basename(f))}
test_subject_folders_to_process = [
    f for f in all_test_subject_folders
    if extract_numeric_id(os.path.basename(f)) not in train_ids and not f.endswith(".ipynb_checkpoints")
]

print(f"Starting predictions for {len(test_subject_folders_to_process)} subjects in '{base_dir_for_test_subjects}'...")
for subject_path in test_subject_folders_to_process:
    try:
        predict_universeg_3d(subject_path, support_images, support_labels, model, batch_size=1)
    except Exception as e:
        print(f"❌ Failed on {os.path.basename(subject_path)}: {e}")