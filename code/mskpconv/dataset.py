import glob
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config
from preprocess_atl03_features import load_and_preprocess_single_csv


def split_files(data_dir: str, seed: int = Config.SEED, is_test_split: bool = True):
    """Split CSV files under one directory into train/val/test lists."""
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    assert len(all_files) > 0, f"No CSV files found in {data_dir}"

    rng = random.Random(seed)
    rng.shuffle(all_files)

    total_files = len(all_files)
    if is_test_split:
        n_train = int(total_files * Config.TRAIN_RATIO)
        n_val = int(total_files * Config.VAL_RATIO)
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]
    else:
        total_ratio = Config.TRAIN_RATIO + Config.VAL_RATIO
        n_train = int(total_files * (Config.TRAIN_RATIO / total_ratio))
        train_files = all_files[:n_train]
        val_files = all_files[n_train:]
        test_files = []

    return train_files, val_files, test_files


class PhotonPointCloudDataset(Dataset):
    """Point-cloud dataset for training/validation with optional HDF5 cache."""

    def __init__(
        self,
        file_list: list,
        h5_cache_path: str = None,
        num_points: int = Config.NUM_POINTS,
        repeat: int = 1,
        augment: bool = False,
    ):
        """Load and chunk files into fixed-size samples."""
        self.num_points = num_points
        self.augment = augment
        self.segments = []

        use_hdf5 = Config.USE_HDF5_CACHE and h5_cache_path and os.path.exists(h5_cache_path)
        h5_file = h5py.File(h5_cache_path, "r") if use_hdf5 else None

        for fp in file_list:
            fname = os.path.basename(fp)
            try:
                if use_hdf5 and fname in h5_file:
                    group = h5_file[fname]
                    data = {
                        "num_points": group.attrs.get("num_points", len(group["labels"])),
                        "x_norm": group["x_norm"][:],
                        "z_norm": group["z_norm"][:],
                        "features": group["features"][:],
                        "labels": group["labels"][:],
                    }
                else:
                    data = load_and_preprocess_single_csv(fp)

                n = data["num_points"]
                sort_idx = np.argsort(data["x_norm"])
                step = int(self.num_points * 0.8)

                for start in range(0, n, step):
                    end = start + self.num_points
                    if end <= n:
                        chunk_idx = sort_idx[start:end]
                    elif n >= self.num_points:
                        chunk_idx = sort_idx[n - self.num_points:n]
                    else:
                        chunk_idx = np.random.choice(sort_idx, self.num_points, replace=True)

                    segment = {
                        "x": data["x_norm"][chunk_idx],
                        "z": data["z_norm"][chunk_idx],
                        "features": data["features"][chunk_idx],
                        "labels": data["labels"][chunk_idx],
                    }
                    for _ in range(repeat):
                        self.segments.append(segment)
            except Exception as e:
                print(f"Skip file {fname}: {e}")

        if h5_file is not None:
            h5_file.close()

    def __len__(self):
        """Return number of chunked samples."""
        return len(self.segments)

    def __getitem__(self, idx):
        """Return one sample as tensors: position, features, labels."""
        seg = self.segments[idx]
        pos = np.stack([seg["x"], seg["z"]], axis=-1)
        return (
            torch.from_numpy(pos).float(),
            torch.from_numpy(seg["features"]).float(),
            torch.from_numpy(seg["labels"]).long(),
        )

class PhotonFileTestDataset(Dataset):
    """Dataset for file-wise test-time voting with overlap sampling."""

    def __init__(self, file_list: list, num_points: int = Config.NUM_POINTS):
        """Load test files and build overlapped chunk index mapping."""
        self.num_points = num_points
        self.file_data = []

        for fp in file_list:
            try:
                data = load_and_preprocess_single_csv(fp)
                sort_idx = np.argsort(data["x_norm"])
                for key in ["x_norm", "z_norm", "x_raw", "z_raw", "features", "labels"]:
                    data[key] = data[key][sort_idx]
                self.file_data.append(data)
            except Exception as e:
                print(f"Skip file {os.path.basename(fp)}: {e}")

        self.samples = []
        for fi, data in enumerate(self.file_data):
            n_total = data["num_points"]
            step = int(self.num_points * 0.5)
            for start in range(0, n_total, step):
                end = start + self.num_points
                if end <= n_total:
                    chunk_idx = np.arange(start, end)
                elif n_total >= self.num_points:
                    chunk_idx = np.arange(n_total - self.num_points, n_total)
                else:
                    chunk_idx = np.random.choice(n_total, self.num_points, replace=True)
                self.samples.append((fi, chunk_idx))

    def __len__(self):
        """Return total number of test chunks."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return one test chunk with file id and point indices for voting."""
        fi, choice = self.samples[idx]
        data = self.file_data[fi]

        pos = np.stack([data["x_norm"][choice], data["z_norm"][choice]], axis=-1)
        return (
            torch.from_numpy(pos).float(),
            torch.from_numpy(data["features"][choice]).float(),
            torch.from_numpy(data["x_raw"][choice]).float(),
            torch.from_numpy(data["z_raw"][choice]).float(),
            torch.from_numpy(data["labels"][choice]).long(),
            fi,
            torch.from_numpy(choice).long(),
        )


def get_dataloaders(train_dir=Config.DATA_DIR, val_dir=Config.VAL_DATA_DIR, test_dir=Config.TEST_DATA_DIR):
    """Create train/val/test dataloaders with optional HDF5 cache acceleration."""
    train_files = glob.glob(os.path.join(train_dir, "*.csv"))
    val_files = glob.glob(os.path.join(val_dir, "*.csv"))
    test_files = glob.glob(os.path.join(test_dir, "*.csv"))

    assert len(train_files) > 0, f"No training files found in {train_dir}"

    train_ds = PhotonPointCloudDataset(train_files, h5_cache_path=Config.HDF5_TRAIN_PATH, repeat=1, augment=True)
    val_ds = PhotonPointCloudDataset(val_files, h5_cache_path=Config.HDF5_VAL_PATH, repeat=1, augment=False)
    test_ds = PhotonPointCloudDataset(test_files, h5_cache_path=Config.HDF5_TEST_PATH, repeat=1, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
