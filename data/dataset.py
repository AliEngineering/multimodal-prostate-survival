import os
import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocess import load_patient_mri, load_patient_mask, preprocess_mri
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class ProstateMultimodalDataset(Dataset):
    def __init__(self, patient_ids, patient_clin, patient_surv, data_root, pad=20):
        self.patient_ids = patient_ids
        self.patient_clin = patient_clin
        self.patient_surv = patient_surv
        self.data_root = data_root
        self.pad = pad

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        # MRI
        vol = load_patient_mri(pid, self.data_root)
        mask = load_patient_mask(pid, self.data_root)
        mri_tensor = preprocess_mri(vol, mask, pad=self.pad)

        # Clinical
        clin = torch.tensor(self.patient_clin[pid], dtype=torch.float32)

        # Survival
        t, e = self.patient_surv[pid]
        t = torch.tensor(t, dtype=torch.float32)
        e = torch.tensor(e, dtype=torch.float32)

        return {
            "mri": mri_tensor,
            "clinical": clin,
            "time": t,
            "event": e
        }


def get_datasets_for_fold(fold_id):
    """
    This expects fold_splits, patient_clin, and patient_surv to be loaded globally.
    """
    from data.preprocess import patient_clin, patient_surv, fold_splits, DATA_ROOT

    train_ids = fold_splits[fold_id]["train"]
    val_ids = fold_splits[fold_id]["val"]

    train_ds = ProstateMultimodalDataset(train_ids, patient_clin, patient_surv, DATA_ROOT)
    val_ds = ProstateMultimodalDataset(val_ids, patient_clin, patient_surv, DATA_ROOT)
    return train_ds, val_ds
