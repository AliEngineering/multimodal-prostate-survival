import os
import json
from glob import glob

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --------- GLOBALS (used by dataset & trainer) ---------
DATA_ROOT = None          # e.g. "/content/drive/MyDrive/Multimodal-Quiz"
patient_clin = {}         # pid -> np.array(features)
patient_surv = {}         # pid -> (time, event)
fold_splits = {}          # fold_id -> {"train": [pids], "val": [pids]}
clin_dim = None           # feature dimension


# ========= MRI / MASK LOADING =========

def _load_mha(path: str) -> np.ndarray:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    return arr.astype(np.float32)


def load_patient_mri(pid, data_root):
    """
    Loads T2, ADC, B1500 for a patient, resamples each to a shared reference
    shape (T2 shape), and returns np array (3, Z, Y, X).
    """
    import SimpleITK as sitk
    import numpy as np
    import os

    t2_path  = os.path.join(data_root, "radiology/mpMRI", pid, f"{pid}_0000.mha")
    adc_path = os.path.join(data_root, "radiology/mpMRI", pid, f"{pid}_0001.mha")
    b15_path = os.path.join(data_root, "radiology/mpMRI", pid, f"{pid}_0002.mha")

    paths = [t2_path, adc_path, b15_path]

    imgs = [sitk.ReadImage(p) for p in paths]

    # --- use T2 as reference space ---
    ref = imgs[0]
    ref_size   = ref.GetSize()
    ref_spacing = ref.GetSpacing()
    ref_origin  = ref.GetOrigin()
    ref_dir     = ref.GetDirection()

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(ref_size)
    resampler.SetOutputSpacing(ref_spacing)
    resampler.SetOutputOrigin(ref_origin)
    resampler.SetOutputDirection(ref_dir)
    resampler.SetInterpolator(sitk.sitkLinear)

    resampled = []
    for img in imgs:
        resampled_img = resampler.Execute(img)
        arr = sitk.GetArrayFromImage(resampled_img)  # (Z, Y, X)
        resampled.append(arr)

    # Stack → shape (3, Z, Y, X)
    vol = np.stack(resampled, axis=0).astype(np.float32)
    return vol


def load_patient_mask(pid, data_root):
    """
    Load prostate mask and resample to T2 reference shape.
    """
    import SimpleITK as sitk
    import numpy as np
    import os

    mask_path = os.path.join(data_root, "radiology/prostate_mask_t2w", f"{pid}_0001_mask.mha")

    # load mask + corresponding T2 image to match shape
    mask_img = sitk.ReadImage(mask_path)
    t2_path = os.path.join(data_root, "radiology/mpMRI", pid, f"{pid}_0000.mha")
    t2_img = sitk.ReadImage(t2_path)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(t2_img.GetSize())
    resampler.SetOutputSpacing(t2_img.GetSpacing())
    resampler.SetOutputOrigin(t2_img.GetOrigin())
    resampler.SetOutputDirection(t2_img.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    mask_resampled = resampler.Execute(mask_img)
    mask_arr = sitk.GetArrayFromImage(mask_resampled).astype(np.uint8)  # (Z, Y, X)

    return mask_arr


def preprocess_mri(vol: np.ndarray, mask: np.ndarray,
                   target_shape=(96, 128, 128), pad: int = 20) -> torch.Tensor:
    """
    vol:  (C, Z, Y, X)
    mask: (Z, Y, X)
    Returns tensor (C, Zt, Yt, Xt)
    """
    C, Z, Y, X = vol.shape
    nz = np.where(mask > 0)

    if len(nz[0]) == 0:
        # no mask at all → use full volume
        zmin, zmax = 0, Z
        ymin, ymax = 0, Y
        xmin, xmax = 0, X
    else:
        zmin, zmax = nz[0].min(), nz[0].max()
        ymin, ymax = nz[1].min(), nz[1].max()
        xmin, xmax = nz[2].min(), nz[2].max()

        zmin = max(zmin - pad, 0)
        ymin = max(ymin - pad, 0)
        xmin = max(xmin - pad, 0)

        zmax = min(zmax + pad, Z)
        ymax = min(ymax + pad, Y)
        xmax = min(xmax + pad, X)

        #  check: if any dimension collapses or is too small,
        #    fall back to the full volume
        if (zmax <= zmin + 1) or (ymax <= ymin + 1) or (xmax <= xmin + 1):
            zmin, zmax = 0, Z
            ymin, ymax = 0, Y
            xmin, xmax = 0, X

    vol_crop = vol[:, zmin:zmax, ymin:ymax, xmin:xmax]  # (C, Zc, Yc, Xc)

    vol_t = torch.from_numpy(vol_crop).unsqueeze(0)      # (1, C, Zc, Yc, Xc)
    vol_t = F.interpolate(
        vol_t,
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)                                        # (C, Zt, Yt, Xt)

    # per-channel z-score ignoring zeros
    for c in range(vol_t.shape[0]):
        ch = vol_t[c]
        nzch = ch[ch != 0]
        if nzch.numel() > 20:
            mean = nzch.mean()
            std = nzch.std()
            if std > 0:
                vol_t[c] = (ch - mean) / std
    return vol_t.float()


# ========= CLINICAL / SURVIVAL =========

def load_clinical_table(clinical_dir: str) -> pd.DataFrame:
    """
    Load each <pid>.json into a row with:
      patient_id, time, event, plus extra clinical fields.
    """
    records = []
    time_key = "time_to_follow-up/BCR"
    event_key = "BCR"

    for path in glob(os.path.join(clinical_dir, "*.json")):
        with open(path, "r") as f:
            d = json.load(f)

        pid = os.path.splitext(os.path.basename(path))[0]

        if time_key not in d or event_key not in d:
            raise KeyError(f"Missing {time_key} or {event_key} in {path}")

        t_raw = d[time_key]
        try:
            time_val = float(t_raw)
        except Exception:
            time_val = float(str(t_raw).replace(",", "."))

        e_raw = d[event_key]
        try:
            event_val = int(round(float(e_raw)))
        except Exception:
            raise ValueError(f"Cannot parse BCR value '{e_raw}' in {path}")

        rec = {"patient_id": pid, "time": time_val, "event": event_val}
        for k, v in d.items():
            if k not in [time_key, event_key]:
                rec[k] = v
        records.append(rec)

    return pd.DataFrame(records)


def build_patient_surv_and_clin(
    clinical_df: pd.DataFrame,
    numeric_features=None,
    categorical_features=None,
):
    df = clinical_df.copy()
    df["patient_id"] = df["patient_id"].astype(str)

    # ---- Define feature sets (as before) ----
    if numeric_features is None:
        numeric_features = [
            "age_at_prostatectomy",
            "primary_gleason",
            "secondary_gleason",
            "ISUP",
            "pre_operative_PSA",
            "positive_surgical_margins",
            "tertiary_gleason",
            "BCR_PSA",
        ]
    if categorical_features is None:
        categorical_features = [
            "pT_stage",
            "positive_lymph_nodes",
            "capsular_penetration",
            "invasion_seminal_vesicles",
            "lymphovascular_invasion",
            "earlier_therapy",
        ]

    # keep only columns that exist
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    # CRITICAL: force numeric features to be numeric (non-numeric → NaN)
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Also normalize categorical: always strings
    for col in categorical_features:
        df[col] = df[col].astype(str).replace({"nan": "missing", "None": "missing"})
        df[col] = df[col].replace({"": "missing", " ": "missing"})

    # ---- Preprocessor as before ----
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X = df[numeric_features + categorical_features]
    X_proc = preprocessor.fit_transform(X)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    patient_surv = {}
    patient_clin = {}

    for pid, row, x in zip(df["patient_id"], df.itertuples(index=False), X_proc):
        patient_surv[pid] = (row.time, row.event)
        patient_clin[pid] = np.asarray(x, dtype=np.float32)

    clin_dim = X_proc.shape[1]
    return patient_surv, patient_clin, preprocessor, clin_dim


def load_fold_splits(split_csv_path: str):
    df_split = pd.read_csv(split_csv_path)
    assert {"patient_id", "fold"}.issubset(df_split.columns)
    df_split["patient_id"] = df_split["patient_id"].astype(str)

    folds = {}
    for f in sorted(df_split["fold"].unique()):
        val_ids = df_split[df_split["fold"] == f]["patient_id"].tolist()
        train_ids = df_split[df_split["fold"] != f]["patient_id"].tolist()
        folds[int(f)] = {"train": train_ids, "val": val_ids}
    return folds


def init_data(data_root: str):
    """
    Populate global:
      DATA_ROOT, patient_clin, patient_surv, fold_splits, clin_dim
    """
    global DATA_ROOT, patient_clin, patient_surv, fold_splits, clin_dim

    DATA_ROOT = data_root
    clinical_dir = os.path.join(DATA_ROOT, "clinical_data")
    split_csv = os.path.join(DATA_ROOT, "data_split_5fold.csv")

    print("=== init_data ===")
    print("  clinical_dir:", clinical_dir)
    print("  split_csv   :", split_csv)

    clinical_df = load_clinical_table(clinical_dir)
    print("  clinical_df shape:", clinical_df.shape)

    patient_surv, patient_clin, preproc, clin_dim = build_patient_surv_and_clin(clinical_df)
    print(f"  N patients: {len(patient_surv)}")
    print(f"  clinical feature dim: {clin_dim}")

    fold_splits = load_fold_splits(split_csv)
    for f, v in fold_splits.items():
        tr, va = v["train"], v["val"]
        e_tr = sum(patient_surv[p][1] for p in tr if p in patient_surv)
        e_va = sum(patient_surv[p][1] for p in va if p in patient_surv)
        print(f"  Fold {f}: train N={len(tr)}, events={e_tr}, val N={len(va)}, events={e_va}")

    globals()["patient_surv"] = patient_surv
    globals()["patient_clin"] = patient_clin
    globals()["fold_splits"] = fold_splits
    globals()["clin_dim"] = clin_dim
