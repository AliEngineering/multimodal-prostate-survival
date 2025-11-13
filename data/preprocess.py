import os
import numpy as np
import SimpleITK as sitk
import torch
from glob import glob

# Global vars, used by train_multimodal.py
DATA_ROOT = None
patient_clin = {}
patient_surv = {}
fold_splits = {}


def load_volume(path):
    vol = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(vol)
    return arr.astype(np.float32)


def load_patient_mri(pid, root):
    patient_dir = os.path.join(root, "radiology", "mpMRI", pid)
    paths = sorted(glob(os.path.join(patient_dir, "*.mha")))
    vols = [load_volume(p) for p in paths]
    return np.stack(vols, axis=0)


def load_patient_mask(pid, root):
    mask_dir = os.path.join(root, "radiology", "prostate_mask_t2w")
    path = os.path.join(mask_dir, f"{pid}_0001_mask.mha")
    return load_volume(path)


def preprocess_mri(vol, mask, pad=20):
    """
    vol: (3, Z, Y, X)
    mask: (Z, Y, X)
    """
    mask_idx = np.argwhere(mask > 0)
    zmin, ymin, xmin = mask_idx.min(axis=0)
    zmax, ymax, xmax = mask_idx.max(axis=0)

    zmin, ymin, xmin = max(zmin - pad, 0), max(ymin - pad, 0), max(xmin - pad, 0)
    zmax += pad
    ymax += pad
    xmax += pad

    vol_crop = vol[:, zmin:zmax, ymin:ymax, xmin:xmax]
    vol_crop = (vol_crop - vol_crop.mean()) / (vol_crop.std() + 1e-8)

    vol_tensor = torch.tensor(vol_crop, dtype=torch.float32)
    vol_tensor = torch.nn.functional.interpolate(
        vol_tensor.unsqueeze(0),
        size=(96, 128, 128),
        mode="trilinear",
        align_corners=False
    )
    return vol_tensor.squeeze(0)
