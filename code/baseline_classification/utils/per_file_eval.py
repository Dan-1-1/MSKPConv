"""Per-file inference + KDE post-processing + pooled metrics over val+test.

Workflow per file
-----------------
1. Run preprocess() to get (x_norm, z_norm, features, labels) aligned with
   the original CSV row order.
2. Sliding-window vote inference (50% overlap) yields a prediction per photon.
3. Read the original CSV to get *physical* x (along-track distance, m) and
   z (geoid-corrected height, m); apply KDE post-processing in metres.
4. Append (pred, gt) to the global pool.

Final metrics are computed on the concatenated pool — no per-file averaging.
"""
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from ..dataset import _import_preprocess
from ..metrics import compute_all_metrics, metrics_summary_row
from .postprocess import apply_surface_postprocess


_X_COL_CANDIDATES = ("Along_Track_Distance", "Along_Track_Dist", "along_track_distance")
_Z_COL_CANDIDATES = (
    "Geoid_Corrected_Ortho_Height",
    "geoid_corrected_ortho_height",
    "WGS84_Ellipsoid_Height",
)


def _segment_indices(n: int, num_points: int, overlap_ratio: float = 0.5):
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
def vote_predict_one_file(
    model,
    csv_path: str,
    source_root: str,
    num_points: int,
    num_classes: int,
    local_feature_dims: int,
    device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run sliding-window vote inference on one CSV and return (pred[N], gt[N])
    aligned with the photon order produced by preprocess()."""
    preprocess = _import_preprocess(source_root)
    data = preprocess(csv_path)
    n = int(data["num_points"])
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    pos = np.stack([data["x_norm"], data["z_norm"]], axis=-1).astype(np.float32)
    feats = np.asarray(data["features"], dtype=np.float32)
    if feats.shape[-1] < local_feature_dims:
        pad = np.zeros((feats.shape[0], local_feature_dims - feats.shape[-1]),
                       dtype=np.float32)
        feats = np.concatenate([feats, pad], axis=-1)
    feats = feats[:, :local_feature_dims]

    sort_idx = np.argsort(pos[:, 0])
    pos_sorted = pos[sort_idx]
    feats_sorted = feats[sort_idx]
    votes = np.zeros((n, num_classes), dtype=np.float32)

    for ci in _segment_indices(n, num_points, 0.5):
        p = torch.from_numpy(pos_sorted[ci]).float().unsqueeze(0).to(device)
        f = torch.from_numpy(feats_sorted[ci]).float().unsqueeze(0).to(device)
        logits = model(p, f)  # [1, K, C]
        prob = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        for k_idx, gi in enumerate(ci):
            votes[gi] += prob[k_idx]

    pred_sorted = votes.argmax(axis=1)
    pred = np.empty(n, dtype=np.int64)
    pred[sort_idx] = pred_sorted

    gt = np.asarray(data["labels"], dtype=np.int64)
    return pred, gt


def _resolve_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    cl = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cl:
            return cl[c.lower()]
    raise KeyError(f"None of {candidates} found among {list(df.columns)}")


def _read_xz_from_csv(csv_path: str, n_expected: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read along-track distance and geoid-corrected height from a CSV, in
    metres. Aligns to the first ``min(len(df), n_expected)`` rows."""
    df = pd.read_csv(csv_path)
    x_col = _resolve_col(df, _X_COL_CANDIDATES)
    z_col = _resolve_col(df, _Z_COL_CANDIDATES)
    x = df[x_col].to_numpy(dtype=np.float64)
    z = df[z_col].to_numpy(dtype=np.float64)
    if len(x) != n_expected:
        m = min(len(x), n_expected)
        x, z = x[:m], z[:m]
    return x, z


def evaluate_on_files_pooled(
    model,
    file_list: List[str],
    cfg,
    device,
    apply_postprocess: bool = True,
    surface_band_m: float = 0.5,
    seabed_offset_m: float = 1.0,
    kde_window_m: float = 50.0,
    kde_bw: float = 0.3,
) -> Dict[str, object]:
    """Inference + (optional) KDE post-process + pooled metrics across files."""
    model.eval()
    y_pred_all: List[np.ndarray] = []
    y_true_all: List[np.ndarray] = []
    skipped: List[str] = []

    for fp in file_list:
        try:
            pred, gt = vote_predict_one_file(
                model, fp,
                source_root=cfg.source_root,
                num_points=cfg.num_points,
                num_classes=cfg.num_classes,
                local_feature_dims=cfg.local_feature_dims,
                device=device,
            )
        except Exception as exc:
            skipped.append(f"{os.path.basename(fp)}: predict fail ({exc})")
            continue

        if pred.size == 0:
            skipped.append(f"{os.path.basename(fp)}: empty after preprocess")
            continue

        if apply_postprocess:
            try:
                x_phys, z_phys = _read_xz_from_csv(fp, n_expected=pred.size)
                m = min(len(x_phys), len(pred))
                pred = apply_surface_postprocess(
                    x_phys[:m], z_phys[:m], pred[:m],
                    surface_band_m=surface_band_m,
                    seabed_offset_m=seabed_offset_m,
                    window_m=kde_window_m,
                    bw=kde_bw,
                )
                gt = gt[:m]
            except Exception as exc:
                skipped.append(f"{os.path.basename(fp)}: postprocess skipped ({exc})")
                # Use raw model predictions if post-processing fails.

        y_pred_all.append(pred)
        y_true_all.append(gt)

    if skipped:
        print(f"[per-file eval] {len(skipped)} file(s) had issues:")
        for s in skipped:
            print(f"  - {s}")

    if not y_pred_all:
        empty_metrics = compute_all_metrics(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            cfg.num_classes, cfg.class_names,
        )
        return {
            "test_metrics": empty_metrics,
            "test_summary_row": metrics_summary_row(empty_metrics, cfg.class_names),
            "n_files_used": 0,
            "n_files_total": len(file_list),
            "n_points": 0,
        }

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    metrics = compute_all_metrics(y_true, y_pred, cfg.num_classes, cfg.class_names)
    return {
        "test_metrics": metrics,
        "test_summary_row": metrics_summary_row(metrics, cfg.class_names),
        "n_files_used": len(y_pred_all),
        "n_files_total": len(file_list),
        "n_points": int(y_pred.size),
    }


def collect_val_test_files(cfg) -> List[str]:
    """Return sorted (val + test) CSV files, skipping any Downloaded_ATL24 dir."""
    val_files = sorted(glob.glob(os.path.join(cfg.val_dir, "*.csv")))
    test_files = sorted(glob.glob(os.path.join(cfg.test_dir, "*.csv")))
    return val_files + test_files
