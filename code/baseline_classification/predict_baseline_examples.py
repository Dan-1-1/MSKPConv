"""E3: Single-CSV multi-model prediction comparison.

For the target file
    ``/root/autodl-tmp/data/Predictions_selected/ATL03_..._gt1l_25.1358_25.2600.csv``

1. Load the trained checkpoints produced by E1 for PointNet++ / PointNeXt /
   KPConv / DGCNN / SKPConv.
2. Run sliding-window inference (50% overlap, vote across overlaps).
3. Read the matching ATL24 reference CSV (if available) for an additional
   "ATL24_reference" prediction column.
4. Write a unified CSV with one row per photon and one column per model
   prediction, plus the ground-truth Label.
5. Save a side-by-side scatter plot (along-track x vs height z, coloured by
   class) for visual comparison.
"""
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import config_from_args, ExperimentConfig
from .dataset import _import_preprocess
from .models.factory import build_baseline_model, build_proposed_wrapped
from .utils.atl24_reference import _map_atl24_labels
from .utils.per_file_eval import _read_xz_from_csv
from .utils.postprocess import apply_surface_postprocess
from .utils.seed import set_seed


_LABEL_COL_CANDIDATES = ("labels", "Label", "Labels", "label", "LABEL")


def _find_label_col(df):
    for c in _LABEL_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


CLASS_COLORS = {0: "#a0a0a0", 1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e"}


def _checkpoint_path(save_root: str, model_dir: str) -> str:
    return os.path.join(save_root, "e1_main_comparison", model_dir, "best.pt")


def _load_model(name: str, ckpt_path: str, cfg: ExperimentConfig, device):
    if name == "SKPConv":
        model = build_proposed_wrapped(cfg.source_root, variant="full",
                                        num_classes=cfg.num_classes)
    else:
        key = {"PointNet++": "pointnet2", "PointNeXt": "pointnext",
               "KPConv": "kpconv", "DGCNN": "dgcnn"}[name]
        model = build_baseline_model(key, num_classes=cfg.num_classes,
                                      in_feat=cfg.local_feature_dims, pos_dim=2)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found for {name}: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def _segment_indices(n: int, num_points: int, overlap_ratio: float = 0.5):
    """Return list of np.array index slices covering [0, n) with overlap."""
    if n <= num_points:
        return [np.arange(n)]
    step = max(1, int(num_points * (1.0 - overlap_ratio)))
    out = []
    for start in range(0, n, step):
        end = start + num_points
        if end <= n:
            out.append(np.arange(start, end))
        else:
            out.append(np.arange(n - num_points, n))
            break
    return out


@torch.no_grad()
def _vote_predictions(model, pos: np.ndarray, feats: np.ndarray, num_classes: int,
                       num_points: int, device) -> np.ndarray:
    """Sliding-window vote inference. pos: [N,2], feats: [N,F]. Returns [N] labels."""
    n = pos.shape[0]
    sort_idx = np.argsort(pos[:, 0])
    pos_sorted = pos[sort_idx]
    feats_sorted = feats[sort_idx]
    votes = np.zeros((n, num_classes), dtype=np.float32)

    chunks = _segment_indices(n, num_points, 0.5)
    for ci in chunks:
        p = torch.from_numpy(pos_sorted[ci]).float().unsqueeze(0).to(device)
        f = torch.from_numpy(feats_sorted[ci]).float().unsqueeze(0).to(device)
        logits = model(p, f)  # [1, K, C]
        prob = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        for k_idx, gi in enumerate(ci):
            votes[gi] += prob[k_idx]
    pred_sorted = votes.argmax(axis=1)
    pred = np.empty(n, dtype=np.int64)
    pred[sort_idx] = pred_sorted
    return pred


def _atl24_pred_for_file(target_csv: str, atl24_dir: str = "") -> Optional[np.ndarray]:
    """Read the ATL24 prediction column directly from the target CSV.

    The new data layout has one CSV per file containing both:
        - ``Label``  : ATL24 prediction code (mapped here to unified codes)
        - ``labels`` : manual ground truth (used as the GT column elsewhere)
    """
    try:
        target_df = pd.read_csv(target_csv)
    except Exception as exc:
        print(f"[E3] could not read target CSV for ATL24 reference: {exc}")
        return None
    if "Label" not in target_df.columns:
        print(f"[E3] no 'Label' (ATL24 prediction) column in {target_csv}")
        return None
    pred_raw = target_df["Label"].to_numpy().astype(np.int64)
    return _map_atl24_labels(pred_raw)


def _save_comparison_plot(out_path: str, target_df: pd.DataFrame, pred_columns: List[str]):
    """One scatter subplot per prediction column + ground truth."""
    n_plots = 1 + len(pred_columns)
    cols = 2
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.5 * rows), squeeze=False)
    x = target_df["Along_Track_Distance"] if "Along_Track_Distance" in target_df.columns \
        else target_df.iloc[:, 0]
    z = target_df["Geoid_Corrected_Ortho_Height"] if "Geoid_Corrected_Ortho_Height" in target_df.columns \
        else target_df.iloc[:, 1]

    gt_col = _find_label_col(target_df)
    panels = [("Ground_Truth", target_df[gt_col].values)] if gt_col is not None else []
    for c in pred_columns:
        panels.append((c, target_df[c].values))

    for i, (title, lbl) in enumerate(panels):
        ax = axes[i // cols][i % cols]
        for k, color in CLASS_COLORS.items():
            mask = lbl == k
            if mask.any():
                ax.scatter(x[mask], z[mask], s=1, c=color, label=f"class {k}")
        ax.set_title(title)
        ax.set_xlabel("Along-track distance")
        ax.set_ylabel("Height")
        ax.legend(markerscale=4, fontsize=7, loc="lower right")
    for j in range(len(panels), rows * cols):
        axes[j // cols][j % cols].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def run_e3(cfg: ExperimentConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    target_csv = cfg.pred_single_csv
    if not os.path.exists(target_csv):
        raise FileNotFoundError(f"Target prediction CSV not found: {target_csv}")
    out_dir = cfg.pred_output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Preprocess raw CSV using the project's preprocessing pipeline
    preprocess = _import_preprocess(cfg.source_root)
    data = preprocess(target_csv)
    n = int(data["num_points"])
    pos = np.stack([data["x_norm"], data["z_norm"]], axis=-1).astype(np.float32)
    feats = np.asarray(data["features"], dtype=np.float32)
    if feats.shape[-1] < cfg.local_feature_dims:
        pad = np.zeros((feats.shape[0], cfg.local_feature_dims - feats.shape[-1]), dtype=np.float32)
        feats = np.concatenate([feats, pad], axis=-1)
    feats = feats[:, : cfg.local_feature_dims]

    print(f"[E3] target CSV: {target_csv}  N={n}")

    # Load original CSV to preserve full coordinate columns for output
    target_df = pd.read_csv(target_csv)

    pred_columns: List[str] = []
    model_dirs = {
        "PointNet++": "PointNet++",
        "PointNeXt": "PointNeXt",
        "KPConv": "KPConv",
        "DGCNN": "DGCNN",
        "SKPConv": "Proposed_method",
    }
    for display, dirname in model_dirs.items():
        ckpt = _checkpoint_path(cfg.save_root, dirname)
        try:
            model = _load_model(display, ckpt, cfg, device)
        except Exception as exc:
            print(f"[E3] skip {display}: {exc}")
            continue
        print(f"[E3] inferring with {display} ...")
        pred = _vote_predictions(model, pos, feats,
                                  num_classes=cfg.num_classes,
                                  num_points=cfg.num_points, device=device)
        if cfg.apply_postprocess:
            try:
                x_phys, z_phys = _read_xz_from_csv(target_csv, n_expected=pred.size)
                m = min(len(x_phys), len(pred))
                pred_pp = pred.copy()
                pred_pp[:m] = apply_surface_postprocess(
                    x_phys[:m], z_phys[:m], pred[:m],
                    surface_band_m=cfg.kde_surface_band_m,
                    seabed_offset_m=cfg.kde_seabed_offset_m,
                    window_m=cfg.kde_window_m,
                    bw=cfg.kde_bw,
                )
                pred = pred_pp
            except Exception as exc:
                print(f"[E3] postprocess skipped for {display}: {exc}")
        col = f"Pred_{display.replace('+', 'P').replace('++', 'PP')}"
        target_df[col] = pred
        pred_columns.append(col)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ATL24 reference column
    ref_pred = _atl24_pred_for_file(target_csv, cfg.atl24_dir)
    if ref_pred is not None:
        target_df["Pred_ATL24_reference"] = ref_pred
        pred_columns.append("Pred_ATL24_reference")

    out_csv = os.path.join(out_dir, os.path.basename(target_csv).replace(".csv", "_compared.csv"))
    target_df.to_csv(out_csv, index=False)
    print(f"[E3] unified predictions saved to: {out_csv}")

    plot_path = os.path.join(out_dir, os.path.basename(target_csv).replace(".csv", "_compared.png"))
    _save_comparison_plot(plot_path, target_df, pred_columns)
    print(f"[E3] comparison plot saved to: {plot_path}")


def main():
    cfg = config_from_args()
    run_e3(cfg)


if __name__ == "__main__":
    main()
