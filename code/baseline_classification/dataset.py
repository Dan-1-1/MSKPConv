"""Photon point-cloud dataset for baselines1.

Loads CSV files via the project's preprocessing pipeline, segments tracks into
fixed-size windows, and returns (positions, features, labels) tuples.
"""
import glob
import os
import sys
from typing import List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _import_preprocess(source_root: str):
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from preprocess_atl03_features import load_and_preprocess_single_csv  # type: ignore
    return load_and_preprocess_single_csv


class PhotonSegmentDataset(Dataset):
    """Sliding-window photon segments for training/eval."""

    def __init__(
        self,
        file_list: List[str],
        source_root: str,
        h5_cache_path: Optional[str] = None,
        num_points: int = 4096,
        local_feature_dims: int = 24,
        input_mode: str = "pos_local",
        overlap_ratio: float = 0.2,
    ):
        self.num_points = num_points
        self.local_feature_dims = local_feature_dims
        self.input_mode = input_mode
        self.segments: List[dict] = []

        preprocess = _import_preprocess(source_root)
        use_h5 = bool(h5_cache_path) and os.path.exists(h5_cache_path)
        h5f = h5py.File(h5_cache_path, "r") if use_h5 else None

        step = max(1, int(num_points * (1.0 - overlap_ratio)))

        for fp in file_list:
            fname = os.path.basename(fp)
            try:
                if use_h5 and fname in h5f:
                    g = h5f[fname]
                    data = {
                        "num_points": int(g.attrs.get("num_points", len(g["labels"]))),
                        "x_norm": g["x_norm"][:],
                        "z_norm": g["z_norm"][:],
                        "features": g["features"][:],
                        "labels": g["labels"][:],
                    }
                else:
                    data = preprocess(fp)
                n = int(data["num_points"])
                if n < num_points // 4:
                    continue
                sort_idx = np.argsort(data["x_norm"])
                for start in range(0, n, step):
                    end = start + num_points
                    if end <= n:
                        idx = sort_idx[start:end]
                    elif n >= num_points:
                        idx = sort_idx[n - num_points: n]
                    else:
                        idx = np.random.choice(sort_idx, num_points, replace=True)
                    self.segments.append({
                        "x": data["x_norm"][idx],
                        "z": data["z_norm"][idx],
                        "features": data["features"][idx],
                        "labels": data["labels"][idx],
                    })
                    if end >= n:
                        break
            except Exception as exc:  # pragma: no cover - debug aid
                print(f"[PhotonSegmentDataset] skip {fname}: {exc}")

        if h5f is not None:
            h5f.close()

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        seg = self.segments[idx]
        pos = np.stack([seg["x"], seg["z"]], axis=-1).astype(np.float32)
        feats = seg["features"].astype(np.float32)
        if feats.shape[-1] < self.local_feature_dims:
            pad = np.zeros((feats.shape[0], self.local_feature_dims - feats.shape[-1]), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=-1)
        feats = feats[:, : self.local_feature_dims]
        if self.input_mode == "pos_only":
            feats = np.zeros_like(feats)
        return (
            torch.from_numpy(pos).float(),
            torch.from_numpy(feats).float(),
            torch.from_numpy(seg["labels"]).long(),
        )


def build_loaders(cfg, input_mode: str = "pos_local"):
    train_files = sorted(glob.glob(os.path.join(cfg.train_dir, "*.csv")))
    val_files = sorted(glob.glob(os.path.join(cfg.val_dir, "*.csv")))
    test_files = sorted(glob.glob(os.path.join(cfg.test_dir, "*.csv")))

    common = dict(
        source_root=cfg.source_root,
        num_points=cfg.num_points,
        local_feature_dims=cfg.local_feature_dims,
        input_mode=input_mode,
    )
    train_ds = PhotonSegmentDataset(train_files, h5_cache_path=cfg.h5_train, overlap_ratio=0.2, **common)
    val_ds = PhotonSegmentDataset(val_files, h5_cache_path=cfg.h5_val, overlap_ratio=0.5, **common)
    test_ds = PhotonSegmentDataset(test_files, h5_cache_path=cfg.h5_test, overlap_ratio=0.5, **common)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        drop_last=True, num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
