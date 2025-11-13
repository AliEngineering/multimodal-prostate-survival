import torch
from torch.utils.data import Dataset

from .preprocess import (
    load_patient_mri,
    load_patient_mask,
    preprocess_mri,
    DATA_ROOT,
    patient_clin,
    patient_surv,
    fold_splits,
)


class ProstateMultimodalDataset(Dataset):
    """
    Dataset returning:
      {
        "pid": str,
        "mri": (C, Z, Y, X) tensor,
        "clinical": (F,) tensor,
        "time": scalar,
        "event": scalar
      }
    """
    def __init__(self, patient_ids, pad=20, target_shape=(96, 128, 128)):
        self.patient_ids = [
            str(pid)
            for pid in patient_ids
            if str(pid) in patient_surv and str(pid) in patient_clin
        ]
        self.pad = pad
        self.target_shape = target_shape

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        vol = load_patient_mri(pid, DATA_ROOT)   # (C, Z, Y, X)
        mask = load_patient_mask(pid, DATA_ROOT) # (Z, Y, X)
        vol_pp = preprocess_mri(vol, mask, target_shape=self.target_shape, pad=self.pad)

        clin_vec = patient_clin[pid]
        time, event = patient_surv[pid]

        return {
            "pid": pid,
            "mri": vol_pp.float(),
            "clinical": torch.tensor(clin_vec, dtype=torch.float32),
            "time": torch.tensor(time, dtype=torch.float32),
            "event": torch.tensor(event, dtype=torch.float32),
        }


def get_datasets_for_fold(fold_id: int, pad: int = 20, target_shape=(96, 128, 128)):
    """
    Uses global fold_splits from preprocess.init_data().
    """
    if fold_id not in fold_splits:
        raise KeyError(f"Fold {fold_id} not found in fold_splits (did you call init_data?)")

    train_ids = fold_splits[fold_id]["train"]
    val_ids = fold_splits[fold_id]["val"]

    train_ds = ProstateMultimodalDataset(train_ids, pad=pad, target_shape=target_shape)
    val_ds = ProstateMultimodalDataset(val_ids, pad=pad, target_shape=target_shape)

    return train_ds, val_ds
