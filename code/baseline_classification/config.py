"""Unified configuration for baselines1 experiments."""
import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


CLASS_NAMES: Tuple[str, ...] = ("noise", "surface", "land", "seabed")
NUM_CLASSES: int = 4
RELEASE_ROOT = Path(__file__).resolve().parents[2]
ATL03_DATA_ROOT = RELEASE_ROOT / "data" / "atl03_photon_classification"
MSKPCONV_CODE_ROOT = RELEASE_ROOT / "code" / "mskpconv"
DEFAULT_SAVE_ROOT = RELEASE_ROOT / "results" / "photon_classification" / "baseline_classification"


@dataclass
class ExperimentConfig:
    # Paths
    data_root: str = str(ATL03_DATA_ROOT)
    source_root: str = str(MSKPCONV_CODE_ROOT)
    save_root: str = str(DEFAULT_SAVE_ROOT)
    train_dir: str = ""
    val_dir: str = ""
    test_dir: str = ""
    atl24_dir: str = ""
    h5_train: str = ""
    h5_val: str = ""
    h5_test: str = ""
    pred_single_csv: str = ""
    pred_output_dir: str = ""

    # Training hyper-parameters
    device: str = "cuda:0"
    num_points: int = 4096
    batch_size: int = 4
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 5e-4
    num_workers: int = 0
    seed: int = 42

    # Data dimensions
    local_feature_dims: int = 24
    num_classes: int = NUM_CLASSES
    class_names: Tuple[str, ...] = CLASS_NAMES

    # KDE post-processing (operates in physical metres)
    apply_postprocess: bool = True
    kde_surface_band_m: float = 0.5
    kde_seabed_offset_m: float = 1.0
    kde_window_m: float = 50.0
    kde_bw: float = 0.3

    # Final-evaluation behaviour: pool predictions across val+test (50 files)
    eval_on_val_test: bool = True

    def finalize(self) -> "ExperimentConfig":
        self.train_dir = self.train_dir or os.path.join(self.data_root, "train")
        self.val_dir = self.val_dir or os.path.join(self.data_root, "validation")
        self.test_dir = self.test_dir or os.path.join(self.data_root, "test")
        self.atl24_dir = self.atl24_dir or os.path.join(self.test_dir, "Downloaded_ATL24")
        cache_dir = os.path.join(self.data_root, "cache")
        self.h5_train = self.h5_train or os.path.join(cache_dir, "train_data.h5")
        self.h5_val = self.h5_val or os.path.join(cache_dir, "val_data.h5")
        self.h5_test = self.h5_test or os.path.join(cache_dir, "test_data.h5")
        self.pred_output_dir = self.pred_output_dir or os.path.join(self.save_root, "predictions")
        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(self.pred_output_dir, exist_ok=True)
        return self


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ATL03_DATA_ROOT))
    parser.add_argument("--source-root", default=str(MSKPCONV_CODE_ROOT))
    parser.add_argument("--save-root", default=str(DEFAULT_SAVE_ROOT))
    parser.add_argument("--train-dir", default="")
    parser.add_argument("--val-dir", default="")
    parser.add_argument("--test-dir", default="")
    parser.add_argument("--atl24-dir", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-feature-dims", type=int, default=24)
    parser.add_argument(
        "--no-postprocess", dest="apply_postprocess", action="store_false",
        help="Disable KDE post-processing on model predictions.",
    )
    parser.set_defaults(apply_postprocess=True)
    parser.add_argument("--kde-surface-band-m", type=float, default=0.5,
                        help="±band (m) around estimated sea surface to relabel as surface.")
    parser.add_argument("--kde-seabed-offset-m", type=float, default=1.0,
                        help="Depth (m) below sea surface to relabel signal as seabed.")
    parser.add_argument("--kde-window-m", type=float, default=50.0,
                        help="Sliding-window half-width (m) for KDE sea-surface estimation.")
    parser.add_argument("--kde-bw", type=float, default=0.3,
                        help="KDE bandwidth in z (m).")
    parser.add_argument(
        "--test-only-eval", dest="eval_on_val_test", action="store_false",
        help="Evaluate models on the test set only (default pools val+test).",
    )
    parser.set_defaults(eval_on_val_test=True)
    return parser


def config_from_args(args=None) -> ExperimentConfig:
    parser = build_argparser()
    ns = parser.parse_args(args=args)
    return ExperimentConfig(**vars(ns)).finalize()
