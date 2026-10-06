"""Configuration for the reproducible MSKPConv pipeline.

All paths are relative to the repository by default. Set the environment
variables documented below when running on another machine or with a separate
output directory; no server-specific absolute paths are required.
"""

import json
import os
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MSKPCONV_DATA_ROOT", RELEASE_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("MSKPCONV_OUTPUT_ROOT", RELEASE_ROOT / "results"))
MODEL_ROOT = Path(os.environ.get("MSKPCONV_MODEL_ROOT", RELEASE_ROOT / "models"))


class Config:
    DATA_DIR = str(DATA_ROOT / "Train")
    VAL_DATA_DIR = str(DATA_ROOT / "Val")
    TEST_DATA_DIR = str(DATA_ROOT / "Test")

    MODEL_SAVE_DIR = str(MODEL_ROOT / "mskpconv")
    PREDICTION_SAVE_DIR = str(OUTPUT_ROOT / "photon_classification")
    SAVE_DIR_PLOT = PREDICTION_SAVE_DIR
    SAVE_DIR = str(OUTPUT_ROOT / "logs")

    USE_HDF5_CACHE = True
    CACHE_DIR = str(DATA_ROOT / "cache")
    HDF5_TRAIN_PATH = os.path.join(CACHE_DIR, "train_data.h5")
    HDF5_VAL_PATH = os.path.join(CACHE_DIR, "val_data.h5")
    HDF5_TEST_PATH = os.path.join(CACHE_DIR, "test_data.h5")

    DEVICE = os.environ.get("MSKPCONV_DEVICE", "cuda:0")

    LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
    NUM_CLASSES = 4
    CLASS_NAMES = ["noise", "surface", "land", "seabed"]

    COL_X = "Along_Track_Dist"
    COL_Z = ["Geoid_Corrected_Ortho_Height"]
    COL_LABEL = "labels"

    LOCAL_FEATURE_DIMS = 24
    MODEL_INPUT_DIMS = LOCAL_FEATURE_DIMS + 2
    NUM_POINTS = 4096

    # The four anisotropic scales and neighborhood sizes used by the 24-D
    # handcrafted feature extractor.
    KNN_CONFIGS = [
        (5, 1 / 5, 1.0),
        (10, 1 / 10, 2.0),
        (15, 1 / 15, 2.0),
        (25, 1 / 20, 4.0),
    ]
    DENSITY_CONFIGS = [
        (1.0, 1 / 5, 1.0),
        (2.0, 1 / 10, 2.0),
        (4.0, 1 / 15, 2.0),
        (5.0, 1 / 20, 4.0),
    ]
    FEATURE_CONFIG_PATH = "default"

    K_NEIGHBORS = 20
    INITIAL_RADIUS = 3
    DIST_SCALE_X = 1.0 / 5
    DIST_SCALE_Z = 3.0
    EMB_DIMS = 256
    DROPOUT = 0.3

    PHYSICAL_LOSS = False
    AUTO_PREDICT_REGIONS = False
    PREDICT_REGION_DIRS = []

    # "fps", "hybrid", "stride", or "density_fps_stride".
    DOWNSAMPLE_STRATEGY = "stride"
    DENSITY_GRID_SIZE = 32
    DENSITY_AWARE_ALPHA = 1.0

    BATCH_SIZE = 4
    EPOCHS = 200
    LR = 1e-4
    WEIGHT_DECAY = 5e-4
    WARMUP_EPOCHS = 10
    SCHEDULER_T0 = 150
    EARLY_STOP_PATIENCE = 200

    SEED = 42
    NUM_WORKERS = 0
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.2


def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _apply_runtime_overrides():
    """Apply optional feature, seed, cache, and output overrides."""
    feature_config_path = os.environ.get("FEATURE_CONFIG_PATH")
    if feature_config_path:
        with open(feature_config_path, "r", encoding="utf-8") as handle:
            feature_config = json.load(handle)
        if "knn_configs" in feature_config:
            Config.KNN_CONFIGS = [tuple(item) for item in feature_config["knn_configs"]]
        if "density_configs" in feature_config:
            Config.DENSITY_CONFIGS = [tuple(item) for item in feature_config["density_configs"]]
        Config.FEATURE_CONFIG_PATH = feature_config_path

    experiment_root = os.environ.get("EXPERIMENT_ROOT")
    if experiment_root:
        experiment_root = Path(experiment_root)
        Config.MODEL_SAVE_DIR = str(experiment_root / "model")
        Config.PREDICTION_SAVE_DIR = str(experiment_root / "prediction")
        Config.SAVE_DIR_PLOT = Config.PREDICTION_SAVE_DIR
        Config.SAVE_DIR = str(experiment_root / "log")
        Config.CACHE_DIR = str(experiment_root / "cache")
        Config.HDF5_TRAIN_PATH = os.path.join(Config.CACHE_DIR, "train_data.h5")
        Config.HDF5_VAL_PATH = os.path.join(Config.CACHE_DIR, "val_data.h5")
        Config.HDF5_TEST_PATH = os.path.join(Config.CACHE_DIR, "test_data.h5")

    Config.SEED = int(os.environ.get("EXPERIMENT_SEED", Config.SEED))
    Config.AUTO_PREDICT_REGIONS = _env_flag("AUTO_PREDICT_REGIONS", Config.AUTO_PREDICT_REGIONS)
    Config.USE_HDF5_CACHE = _env_flag("USE_HDF5_CACHE", Config.USE_HDF5_CACHE)
    if os.environ.get("SENSITIVITY_EPOCHS"):
        Config.EPOCHS = int(os.environ["SENSITIVITY_EPOCHS"])
    if os.environ.get("SENSITIVITY_EARLY_STOP_PATIENCE"):
        Config.EARLY_STOP_PATIENCE = int(os.environ["SENSITIVITY_EARLY_STOP_PATIENCE"])

    for path in (Config.MODEL_SAVE_DIR, Config.PREDICTION_SAVE_DIR, Config.SAVE_DIR, Config.CACHE_DIR):
        Path(path).mkdir(parents=True, exist_ok=True)


_apply_runtime_overrides()
