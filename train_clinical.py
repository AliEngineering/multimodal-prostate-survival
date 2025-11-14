import os
import argparse
import numpy as np
import torch
import random
import pandas as pd

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.dataset import get_datasets_for_fold
from data import preprocess
from utils.survival_loss import cox_ph_loss
from utils.metrics import concordance_index
from utils.wandb_utils import setup_wandb, log_metrics, finish_wandb
from models.clinical_survnet import ClinicalSurvNet  


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders_for_fold(fold_id, batch_size, device, pad):
    train_ds, val_ds = get_datasets_for_fold(fold_id, pad=pad)
    pin = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )
    return train_loader, val_loader


def count_events(loader):
    n = 0
    e = 0
    for batch in loader:
        n += batch["event"].numel()
        e += int(batch["event"].sum().item())
    return n, e


def train_one_fold(fold_id, args, device):
    print(f"\n========== Fold {fold_id} (CLINICAL ONLY) ==========")

    train_loader, val_loader = get_dataloaders_for_fold(
        fold_id, args.batch_size, device, args.pad
    )

    cdim = preprocess.clin_dim
    print(f"Clinical feature dim: {cdim}")

    model = ClinicalSurvNet(
        clin_dim=cdim,
        hidden=64,
        fusion_hidden=128,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    n_train, e_train = count_events(train_loader)
    n_val, e_val = count_events(val_loader)
    print(f"Train N={n_train}, events={e_train}, Val N={n_val}, events={e_val}")

    run = setup_wandb(fold_id, args)

    best_cindex = -1.0
    best_state = None
    all_val_preds = []

    for epoch in range(1, args.epochs + 1):
        # -------- TRAIN --------
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"[Clin] Fold {fold_id} - Epoch {epoch} [train]"):
            clin = batch["clinical"].to(device)  # <--- ONLY CLINICAL
            time = batch["time"].to(device)
            event = batch["event"].to(device)

            optimizer.zero_grad()
            risk = model(clin)                  # <--- ONLY CLINICAL
            loss = cox_ph_loss(risk, time, event)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # -------- VAL --------
        model.eval()
        val_losses = []
        all_risk, all_time, all_event = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Clin] Fold {fold_id} - Epoch {epoch} [val]"):
                clin = batch["clinical"].to(device)
                time = batch["time"].to(device)
                event = batch["event"].to(device)

                risk = model(clin)

                pids = batch["pid"]
                for i in range(risk.shape[0]):
                    all_val_preds.append({
                        "fold": int(fold_id),
                        "pid": str(pids[i]),
                        "time": float(time[i].cpu().item()),
                        "event": int(event[i].cpu().item()),
                        "risk": float(risk[i].detach().cpu().item()),
                        "epoch": int(epoch),
                    })

                loss = cox_ph_loss(risk, time, event)

                val_losses.append(loss.item())
                all_risk.append(risk)
                all_time.append(time)
                all_event.append(event)

        all_risk = torch.cat(all_risk)
        all_time = torch.cat(all_time)
        all_event = torch.cat(all_event)

        c_index = concordance_index(all_risk, all_time, all_event)
        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))

        print(
            f"[Clin Fold {fold_id}] Epoch {epoch} | "
            f"Train {mean_train:.4f} | Val {mean_val:.4f} | C-index {c_index:.4f}"
        )

        log_metrics(epoch, mean_train, mean_val, c_index, fold_id)

        if c_index > best_cindex:
            best_cindex = c_index
            best_state = model.state_dict()
            model_dir = os.path.join(args.out_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            path = os.path.join(model_dir, f"clinical_survnet_fold{fold_id}.pth")
            torch.save(best_state, path)
            print(f"Saved best CLINICAL model for fold {fold_id}: {c_index:.4f} to {path}")

    # save predictions
    if len(all_val_preds) > 0:
        preds_df = pd.DataFrame(all_val_preds)
        os.makedirs(args.out_dir, exist_ok=True)
        csv_path = os.path.join(args.out_dir, f"predictions_clinical_fold{fold_id}.csv")
        preds_df.to_csv(csv_path, index=False)
        print(f"Saved clinical-only fold {fold_id} predictions to {csv_path}")

    finish_wandb(best_cindex)
    return best_cindex


def train_all_folds(args):
    preprocess.init_data(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    fold_ids = [0, 1, 2, 3, 4]
    results = {}

    for f in fold_ids:
        results[f] = train_one_fold(f, args, device)

    print("\n===== FINAL 5-FOLD RESULTS (CLINICAL ONLY) =====")
    for f, c in results.items():
        print(f"Fold {f}: {c:.4f}")
    print(
        f"Mean: {np.mean(list(results.values())):.4f} ± "
        f"{np.std(list(results.values())):.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)   # match main if you want
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--pad", type=int, default=40)      # still used by dataset but MRI ignored
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs_clinical")

    args = parser.parse_args()
    set_seed(42)
    print("Args:", vars(args))
    train_all_folds(args)
