# Multimodal Prostate Cancer Survival Prediction  
*A Reproducible MICCAI-Style Implementation*

This repository contains the complete implementation, training pipeline, and pretrained models for a multimodal deep learning system that predicts prostate cancer survival from **multi-sequence MRI** and **clinical features**.

The project includes:
- **Main Multimodal Survival Model** (MRI + clinical features)  
- **MRI-only model**  
- **Clinical-only model**  

All model weights (5-fold CV), prediction CSVs, and reproducibility artifacts are provided on Hugging Face.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Data Description](#data-description)  
3. [Repository Structure](#repository-structure)  
4. [Environment & Dependencies](#environment--dependencies)  
5. [Model Architectures](#model-architectures)  
6. [Training & Evaluation Pipeline](#training--evaluation-pipeline)  
7. [Instructions to Reproduce](#instructions-to-reproduce)  
8. [Pretrained Models (Hugging Face)](#pretrained-models-hugging-face)  
9. [Ablation Studies](#ablation-studies)  
10. [Reproducibility Checklist (MICCAI)](#reproducibility-checklist-miccai)  
11. [Citation](#citation)

---

## Project Overview

This work develops and evaluates a **multimodal survival prediction model** integrating:

- Three MRI modalities (`ADC.mha`, `ADC_challenge.mha`, `T2w.mha`)  
- Clinical features stored as JSON  
- Time-to-event survival labels  
- Cox proportional hazards loss  

We perform **5-fold cross-validation** and provide:
- Per-case predictions (CSV)
- Best model checkpoint per fold (.pth)
- All results for multimodal / MRI-only / clinical-only models

---

## Data Description

⚠️ **Dataset is private and not included.**

Your dataset folder must follow:

```
Multimodal-Quiz/
│
├── radiology/
│   ├── Patient_001/
│   │    ├── ADC.mha
│   │    ├── ADC_challenge.mha
│   │    └── T2w.mha
│   └── ...
│
├── clinical_data/
│   ├── Patient_001.json
│   ├── Patient_002.json
│   └── ...
│
└── data_split_5fold.csv
```

Clinical features → 29 total after preprocessing  
Labels → `time`, `event`

---

## Repository Structure

```
multimodal-prostate-survival/
│
├── train_multimodal.py
├── train_mri_only.py
├── train_clinical_only.py
│
├── models/
│   └── multimodal_survnet.py
│
├── data/
│   ├── dataset.py
│   └── preprocess.py
│
├── utils/
│   ├── survival_loss.py
│   ├── metrics.py
│   └── wandb_utils.py
│
└── README.md
```

---

## Environment & Dependencies

### Hardware
```
Google Colab Pro
GPU: NVIDIA T4
```

### Python
```
Python 3.12.12
```

### Core packages
```
torch==2.8.0+cu126
torchvision==0.23.0+cu126
torchaudio==2.8.0+cu126
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
nibabel==5.3.2
wandb==0.22.x
tqdm
```

---

## Model Architectures

### Multimodal Model (Main)
- MRI encoder: pretrained **R3D-18**
- Clinical MLP: LayerNorm + ReLU
- Fusion: concat → MLP → risk  
- Loss: Cox proportional hazards  

### MRI-Only Model
- Only MRI branch  
- No clinical features  

### Clinical-Only Model
- MLP on 29 features  
- No MRI encoder  

---

## Training & Evaluation Pipeline

All 3 models use:

```
epochs = 30
batch_size = 2
lr = 3e-5
weight_decay = 1e-5
pad = 40
freeze_mri = True
```

Per-fold outputs:
```
outputs/models/*.pth
outputs/predictions_fold*.csv
```

Each CSV contains:
```
pid, time, event, risk, epoch, fold
```

---

## Instructions to Reproduce

### Train Multimodal Model
```
python train_multimodal.py \
  --data_root "/PATH/Multimodal-Quiz" \
  --epochs 30 \
  --batch_size 2 \
  --lr 3e-5 \
  --weight_decay 1e-5 \
  --pad 40 \
  --freeze_mri \
  --wandb \
  --out_dir "outputs"
```

### Train MRI-Only
```
python train_mri_only.py \
  --data_root "/PATH/Multimodal-Quiz" \
  --epochs 30 \
  --batch_size 2 \
  --lr 3e-5 \
  --weight_decay 1e-5 \
  --pad 40 \
  --wandb \
  --out_dir "outputs_mri"
```

### Train Clinical-Only
```
python train_clinical_only.py \
  --data_root "/PATH/Multimodal-Quiz" \
  --epochs 30 \
  --batch_size 2 \
  --lr 3e-5 \
  --weight_decay 1e-5 \
  --wandb \
  --out_dir "outputs_clinical"
```

---

## Pretrained Models (Hugging Face)

- **Main**  
  https://huggingface.co/dr3mar/prostate-multimodal-survival-main

- **MRI-Only**  
  https://huggingface.co/dr3mar/prostate-multimodal-survival-mri

- **Clinical-Only**  
  https://huggingface.co/dr3mar/prostate-multimodal-survival-clinical

---

## Ablation Study Results

### Multimodal  
**0.8152 ± 0.1189**

### MRI-Only  
**0.6470 ± 0.0759**

### Clinical-Only  
**0.7967 ± 0.1067**

---

## Reproducibility Checklist (MICCAI)

- ✔ Full code released  
- ✔ All hyperparameters listed  
- ✔ Full architecture descriptions  
- ✔ Data splits provided  
- ✔ Preprocessing fully described  
- ✔ Logging with W&B  
- ✔ Deterministic seed  
- ✔ Pretrained weights released  

---

## Citation

```
Ammar, A. (2025).
Multimodal MRI + Clinical Survival Model for Prostate Cancer.
https://github.com/AliEngineering/multimodal-prostate-survival
```
