import torch
from torch.utils.data import Dataset

from . import preprocess  # import the module, not the variables


class ProstateMultimodalDataset(Dataset):
    """
    Each item:
      {
        "pid": str,
        "mri": (C, Z, Y, X) tensor,
        "clinical": (F,) tensor,
        "time": scalar,
        "event": scalar
      }
    """
    def __init__(self, patient_ids, pad=20, target_shape=(96, 128, 128)):
        self.pad = pad
        self.target_shape = target_shape

        # keep only pids that have both clinical and survival
        self.patient_ids = [
            str(pid)
            for pid in patient_ids
            if str(pid) in preprocess.patient_surv
            and str(pid) in preprocess.patient_clin
        ]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        # use functions + globals from preprocess module
        vol = preprocess.load_patient_mri(pid, preprocess.DATA_ROOT)   # (C, Z, Y, X)
        mask = preprocess.load_patient_mask(pid, preprocess.DATA_ROOT) # (Z, Y, X)
        vol_pp = preprocess.preprocess_mri(
            vol,
            mask,
            target_shape=self.target_shape,
            pad=self.pad,
        )  # tensor (C, Zt, Yt, Xt)

        clin_vec = preprocess.patient_clin[pid]
        time, event = preprocess.patient_surv[pid]

        return {
            "pid": pid,
            "mri": vol_pp,
            "clinical": torch.tensor(clin_vec, dtype=torch.float32),
            "time": torch.tensor(time, dtype=torch.float32),
            "event": torch.tensor(event, dtype=torch.float32),
        }


def get_datasets_for_fold(fold_id: int, pad: int = 20, target_shape=(96, 128, 128)):
    """
    Uses global preprocess.fold_splits, which must be populated by init_data(data_root).
    """
    fs = preprocess.fold_splits
    if fold_id not in fs:
        raise KeyError(f"Fold {fold_id} not found in fold_splits. Did you call init_data()?")

    train_ids = fs[fold_id]["train"]
    val_ids = fs[fold_id]["val"]

    train_ds = ProstateMultimodalDataset(train_ids, pad=pad, target_shape=target_shape)
    val_ds = ProstateMultimodalDataset(val_ids, pad=pad, target_shape=target_shape)

    return train_ds, val_ds
