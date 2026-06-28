import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config


COLOR_MAP = {0: "#AAAAAA", 1: "#3498DB", 2: "#27AE60", 3: "#E74C3C"}
LABEL_NAMES = {0: "Noise", 1: "Surface", 2: "Land", 3: "Seabed"}


def save_predictions_to_csv(file_data_list, file_votes, test_dir, save_dir):
    """Write per-file prediction labels to CSV files in a separate output folder."""
    os.makedirs(save_dir, exist_ok=True)

    for fi, data in enumerate(file_data_list):
        base_name = os.path.basename(data["filename"])
        orig_filepath = os.path.join(test_dir, base_name)

        if not os.path.exists(orig_filepath):
            print(f"Missing source file: {orig_filepath}")
            continue

        try:
            df = pd.read_csv(orig_filepath)
        except Exception as e:
            print(f"Read failed: {base_name} -> {e}")
            continue

        final_pred_labels = np.argmax(file_votes[fi], axis=1)
        if len(df) != len(final_pred_labels):
            print(f"Length mismatch: {base_name} ({len(df)} vs {len(final_pred_labels)})")
            continue

        df["pred_label"] = final_pred_labels
        save_path = os.path.join(save_dir, base_name)
        df.to_csv(save_path, index=False)


def plot_training_curves(history: dict, save_dir=Config.SAVE_DIR_PLOT):
    """Plot and save training curves for loss, accuracy, and learning rate."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax.plot(epochs, history["val_loss"], "r-o", markersize=3, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["focal_loss"], "g-o", markersize=3, label="Focal Loss")
    ax.plot(epochs, history["lovasz_loss"], "m-o", markersize=3, label="Lovasz Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc")
    ax.plot(epochs, history["val_acc"], "r-o", markersize=3, label="Val Acc")
    ax.plot(epochs, history["f1"], "g-o", markersize=3, label="F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(epochs, history["lr"], "k-", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(save_dir, "lr_schedule.png")
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(f"Training plots saved to: {save_dir}")


def plot_single_file_result(
    x_raw,
    z_raw,
    true_labels,
    pred_labels,
    filename,
    save_dir=Config.SAVE_DIR_PLOT,
    macro_f1=None,
):
    """Plot one file with ground truth and prediction labels side-by-side."""
    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True, sharey=True)

    for ax, labels, title_suffix in zip(axes, [true_labels, pred_labels], ["Ground Truth", "Prediction"]):
        for cls_id in sorted(COLOR_MAP.keys()):
            mask = labels == cls_id
            if mask.sum() == 0:
                continue
            ax.scatter(
                x_raw[mask],
                z_raw[mask],
                c=COLOR_MAP[cls_id],
                s=0.3,
                alpha=0.6,
                label=f"{LABEL_NAMES[cls_id]} ({mask.sum():,})",
                rasterized=True,
            )

        ax.set_ylabel("Ellipsoid Height (m)")
        ax.set_title(title_suffix)
        ax.legend(fontsize=9, markerscale=10, loc="upper right")
        ax.grid(True, alpha=0.2)

    axes[1].set_xlabel("Along Track Distance (m)")

    acc = (true_labels == pred_labels).mean()
    if macro_f1 is None:
        fig.suptitle(f"{filename}\nAccuracy: {acc:.4f}", y=1.02)
    else:
        fig.suptitle(f"{filename}\nAccuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}", y=1.02)

    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()

    safe_name = os.path.splitext(filename)[0]
    if len(safe_name) > 80:
        safe_name = safe_name[:80]

    path = os.path.join(save_dir, f"{safe_name}_result.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
