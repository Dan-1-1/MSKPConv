"""Training engine: train/eval loop, checkpointing, early stopping."""
import csv
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from .metrics import compute_all_metrics, metrics_summary_row


def run_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    losses = []
    for pos, feats, labels in loader:
        pos = pos.to(device, non_blocking=True)
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(pos, feats)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_pred, y_true = [], []
    for pos, feats, labels in loader:
        pos = pos.to(device, non_blocking=True)
        feats = feats.to(device, non_blocking=True)
        logits = model(pos, feats)
        pred = logits.argmax(dim=-1).cpu().numpy()
        y_pred.append(pred.reshape(-1))
        y_true.append(labels.numpy().reshape(-1))
    if not y_pred:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(y_pred), np.concatenate(y_true)


def train_and_evaluate(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    criterion: nn.Module,
    cfg,
    save_dir: str,
    early_stop_patience: int = 6,
) -> Dict[str, object]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.pt")
    log_path = os.path.join(save_dir, "training_log.csv")

    best_f1 = -1.0
    best_state = None
    no_improve = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_OA", "val_mIoU", "val_Macro-F1"])

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            loss = run_epoch(model, train_loader, optimizer, criterion, device)
            y_pred, y_true = predict(model, val_loader, device)
            metrics = compute_all_metrics(y_true, y_pred, cfg.num_classes, cfg.class_names)
            scheduler.step()

            writer.writerow([epoch, round(loss, 4),
                             round(metrics["OA"], 4),
                             round(metrics["mIoU"], 4),
                             round(metrics["Macro-F1"], 4)])
            f.flush()

            elapsed = time.time() - t0
            print(
                f"[{os.path.basename(save_dir)}] ep{epoch:03d} | "
                f"loss={loss:.4f} | val OA={metrics['OA']:.4f} "
                f"mIoU={metrics['mIoU']:.4f} F1={metrics['Macro-F1']:.4f} | "
                f"{elapsed:.1f}s"
            )

            if metrics["Macro-F1"] > best_f1:
                best_f1 = metrics["Macro-F1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, best_path)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    print(f"  Early stop at epoch {epoch} (no improvement for {early_stop_patience}).")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_pred, y_true = predict(model, test_loader, device)
    test_metrics = compute_all_metrics(y_true, y_pred, cfg.num_classes, cfg.class_names)

    return {
        "model": model,
        "test_metrics": test_metrics,
        "test_summary_row": metrics_summary_row(test_metrics, cfg.class_names),
        "best_val_macro_f1": best_f1,
        "best_ckpt": best_path,
    }
