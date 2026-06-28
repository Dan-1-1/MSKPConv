"""ATL24 reference evaluation by nearest-neighbour matching.

Manual test CSVs live in ``test_dir`` and contain the user labels column
``labels``. ATL24 product CSVs live in ``atl24_dir`` and contain product
labels in ``Label``. Points are matched from each manual photon to its nearest
ATL24 photon using horizontal latitude/longitude distance, then filtered by
height difference. Matched photons are pooled across files for one metric set.

Unified label space:
    0 noise, 1 sea surface, 2 land, 3 seabed

For ATL24 comparison, manual land labels are folded into noise because ATL24
does not identify land as a separate class. Metrics are averaged over the
three evaluated classes: noise, surface, and seabed.
"""
import glob
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..metrics import compute_all_metrics, metrics_summary_row


ATL24_TO_UNIFIED = {
    0: 0,    # unclassified / noise
    40: 3,   # bathymetric seabed
    41: 1,   # sea surface
}

_GT_COL_CANDIDATES = ("labels", "label_manual", "manual_label")
_PRED_COL_CANDIDATES = ("Label", "label", "classification")
_LAT_COL_CANDIDATES = ("latitude", "Latitude", "lat", "Lat")
_LON_COL_CANDIDATES = ("longitude", "Longitude", "lon", "Lon", "lng", "Lng")
_Z_COL_CANDIDATES = (
    "Geoid_Corrected_Ortho_Height",
    "geoid_corrected_ortho_height",
    "WGS84_Ellipsoid_Height",
)
_MAX_HORIZONTAL_DIST_M = 2.0
_MAX_HEIGHT_DIFF_M = 0.5


def _map_atl24_labels(arr: np.ndarray) -> np.ndarray:
    """Map ATL24 product labels to the unified 4-class label space."""
    arr = np.asarray(arr)
    out = np.zeros(arr.shape, dtype=np.int64)
    for k, v in ATL24_TO_UNIFIED.items():
        out[arr == k] = v
    return out


def _map_manual_labels_for_atl24_eval(arr: np.ndarray) -> np.ndarray:
    """Map manual labels for ATL24 comparison, folding land into noise."""
    arr = np.asarray(arr, dtype=np.int64)
    out = arr.copy()
    out[out == 2] = 0
    out[(out < 0) | (out > 3)] = 0
    return out


def _compute_atl24_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    """Compute 4-code metrics but average only noise/surface/seabed classes."""
    class_names = ("noise", "surface", "land", "seabed")
    metrics = compute_all_metrics(y_true, y_pred, 4, class_names)
    eval_classes = ("noise", "surface", "seabed")
    metrics["mIoU"] = float(np.mean([
        metrics["per_class"][name]["iou"] for name in eval_classes
    ]))
    metrics["Macro-F1"] = float(np.mean([
        metrics["per_class"][name]["f1"] for name in eval_classes
    ]))
    return metrics


def _find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    """Exact match first; case-insensitive fallback only as last resort."""
    for c in candidates:
        if c in df.columns:
            return c
    cl = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cl:
            return cl[c.lower()]
    return None


def _match_key(path: str) -> str:
    """Normalize ATL03/ATL24 filenames to a common matching key."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"ATL(?:03|24)_", "ATLXX_", stem, count=1)


def _build_atl24_index(atl24_dir: str) -> Dict[str, str]:
    files = sorted(glob.glob(os.path.join(atl24_dir, "*.csv")))
    out: Dict[str, str] = {}
    for fp in files:
        out.setdefault(_match_key(fp), fp)
    return out


def _scaled_xy(df: pd.DataFrame, lat_col: str, lon_col: str, lat0: float) -> np.ndarray:
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=np.float64)
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=np.float64)
    return np.column_stack([
        lat * 111320.0,
        lon * 111320.0 * math.cos(math.radians(lat0)),
    ])


def _matched_labels(test_fp: str, atl24_fp: str) -> Tuple[np.ndarray, np.ndarray]:
    test_df = pd.read_csv(test_fp)
    atl24_df = pd.read_csv(atl24_fp)

    gt_col = _find_col(test_df, _GT_COL_CANDIDATES)
    pred_col = _find_col(atl24_df, _PRED_COL_CANDIDATES)
    test_lat = _find_col(test_df, _LAT_COL_CANDIDATES)
    test_lon = _find_col(test_df, _LON_COL_CANDIDATES)
    atl24_lat = _find_col(atl24_df, _LAT_COL_CANDIDATES)
    atl24_lon = _find_col(atl24_df, _LON_COL_CANDIDATES)
    test_z = _find_col(test_df, _Z_COL_CANDIDATES)
    atl24_z = _find_col(atl24_df, _Z_COL_CANDIDATES)
    missing = {
        "gt": gt_col, "pred": pred_col,
        "test_lat": test_lat, "test_lon": test_lon,
        "atl24_lat": atl24_lat, "atl24_lon": atl24_lon,
        "test_z": test_z, "atl24_z": atl24_z,
    }
    if any(v is None for v in missing.values()):
        raise KeyError(f"missing required columns: {missing}")

    test_valid = (
        pd.to_numeric(test_df[test_lat], errors="coerce").notna().to_numpy()
        & pd.to_numeric(test_df[test_lon], errors="coerce").notna().to_numpy()
        & pd.to_numeric(test_df[test_z], errors="coerce").notna().to_numpy()
    )
    atl24_valid = (
        pd.to_numeric(atl24_df[atl24_lat], errors="coerce").notna().to_numpy()
        & pd.to_numeric(atl24_df[atl24_lon], errors="coerce").notna().to_numpy()
        & pd.to_numeric(atl24_df[atl24_z], errors="coerce").notna().to_numpy()
    )
    if not test_valid.any() or not atl24_valid.any():
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    test_rows = np.flatnonzero(test_valid)
    atl24_rows = np.flatnonzero(atl24_valid)
    lat0 = float(np.nanmean(np.concatenate([
        pd.to_numeric(test_df.iloc[test_rows][test_lat], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(atl24_df.iloc[atl24_rows][atl24_lat], errors="coerce").to_numpy(dtype=np.float64),
    ])))

    test_xy = _scaled_xy(test_df.iloc[test_rows], test_lat, test_lon, lat0)
    atl24_xy = _scaled_xy(atl24_df.iloc[atl24_rows], atl24_lat, atl24_lon, lat0)
    tree_atl24 = cKDTree(atl24_xy)
    dist_ta, nearest_ta = tree_atl24.query(test_xy, k=1, workers=-1)
    tree_test = cKDTree(test_xy)
    _, nearest_at = tree_test.query(atl24_xy, k=1, workers=-1)

    test_height = pd.to_numeric(
        test_df.iloc[test_rows][test_z], errors="coerce"
    ).to_numpy(dtype=np.float64)
    atl24_height = pd.to_numeric(
        atl24_df.iloc[atl24_rows][atl24_z], errors="coerce"
    ).to_numpy(dtype=np.float64)
    dz = np.abs(test_height - atl24_height[nearest_ta])
    keep = (dist_ta <= _MAX_HORIZONTAL_DIST_M) & (dz <= _MAX_HEIGHT_DIFF_M)
    keep &= nearest_at[nearest_ta] == np.arange(test_xy.shape[0])
    if not keep.any():
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    matched_test_rows = test_rows[keep]
    matched_atl24_rows = atl24_rows[nearest_ta[keep]]
    gt_raw = test_df.iloc[matched_test_rows][gt_col].to_numpy()
    pred_raw = atl24_df.iloc[matched_atl24_rows][pred_col].to_numpy()
    gt = _map_manual_labels_for_atl24_eval(gt_raw)
    pred = _map_atl24_labels(pred_raw)
    return gt, pred


def evaluate_atl24_reference(
    test_dir: str,
    atl24_dir: str = "",
    num_classes: int = 4,
    class_names: Tuple[str, ...] = ("noise", "surface", "land", "seabed"),
) -> Dict[str, object]:
    """Pool ATL24 predictions vs. manual ground truth across matched test CSVs."""
    class_names = ("noise", "surface", "land", "seabed")
    if os.path.basename(os.path.normpath(test_dir)) == "Downloaded_ATL24":
        atl24_dir = atl24_dir or test_dir
        test_dir = os.path.dirname(os.path.normpath(test_dir))
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test dir not found: {test_dir}")
    atl24_dir = atl24_dir or os.path.join(test_dir, "Downloaded_ATL24")
    if not os.path.isdir(atl24_dir):
        raise FileNotFoundError(f"ATL24 dir not found: {atl24_dir}")

    files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))
    if not files:
        raise RuntimeError(f"No CSV files found in {test_dir}")

    atl24_index = _build_atl24_index(atl24_dir)
    y_pred_all: List[np.ndarray] = []
    y_true_all: List[np.ndarray] = []
    used = 0

    for fp in files:
        ref_fp = atl24_index.get(_match_key(fp))
        if ref_fp is None:
            continue
        try:
            gt, pred = _matched_labels(fp, ref_fp)
        except Exception:
            continue
        if pred.size == 0:
            continue
        y_true_all.append(gt)
        y_pred_all.append(pred)
        used += 1

    if not y_pred_all:
        raise RuntimeError(
            "No usable ATL24 reference pairs after filename and lat/lon matching."
        )

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    print(f"[ATL24] files used: {used} / {len(files)} | photons: {int(y_pred.size)}")

    metrics = _compute_atl24_metrics(y_true, y_pred)
    return {
        "test_metrics": metrics,
        "test_summary_row": metrics_summary_row(metrics, class_names),
        "n_files_used": used,
        "n_files_total": len(files),
        "n_points": int(y_pred.size),
    }
