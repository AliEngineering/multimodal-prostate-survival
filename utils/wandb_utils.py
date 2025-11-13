import wandb


def setup_wandb(fold_id, args):
    if not args.wandb:
        return None

    run = wandb.init(
        project="multimodal-prostate-survival",
        name=f"fold_{fold_id}",
        config=vars(args),
        reinit=True
    )
    return run


def log_metrics(epoch, train_loss, val_loss, cindex, fold):
    if wandb.run is not None:
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "c_index": cindex,
            "fold": fold
        })


def finish_wandb(best_cindex):
    if wandb.run is not None:
        wandb.run.summary["best_cindex"] = best_cindex
        wandb.finish()
