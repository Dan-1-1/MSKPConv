import os
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = RELEASE_ROOT / "data" / "atl03_photon_classification"


class Config:
    DATA_DIR = str(DATA_ROOT / "train")
    VAL_DATA_DIR = str(DATA_ROOT / "validation")
    TEST_DATA_DIR = str(DATA_ROOT / "test")

    MODEL_SAVE_DIR = str(RELEASE_ROOT / "models" / "mskpconv")
    PREDICTION_SAVE_DIR = str(RELEASE_ROOT / "results" / "photon_classification")
    SAVE_DIR_PLOT = PREDICTION_SAVE_DIR
    SAVE_DIR = str(RELEASE_ROOT / "results" / "photon_classification" / "logs")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(PREDICTION_SAVE_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    USE_HDF5_CACHE = True
    CACHE_DIR = str(DATA_ROOT / "hdf5_cache")
    HDF5_TRAIN_PATH = os.path.join(CACHE_DIR, "train_data.h5")
    HDF5_VAL_PATH = os.path.join(CACHE_DIR, "val_data.h5")
    HDF5_TEST_PATH = os.path.join(CACHE_DIR, "test_data.h5")

    DEVICE = "cuda:0"

    LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
    NUM_CLASSES = 4
    CLASS_NAMES = ["noise", "surface", "land", "seabed"]

    COL_X = "Along_Track_Dist"
    COL_Z = ["Geoid_Corrected_Ortho_Height"]
    COL_LABEL = "labels"

    LOCAL_FEATURE_DIMS = 24
    MODEL_INPUT_DIMS = LOCAL_FEATURE_DIMS + 2
    NUM_POINTS = 4096

    K_NEIGHBORS = 20
    INITIAL_RADIUS = 3
    DIST_SCALE_X = 1.0 / 5
    DIST_SCALE_Z = 3.0
    EMB_DIMS = 256
    DROPOUT = 0.3

    PHYSICAL_LOSS = False
    AUTO_PREDICT_REGIONS = True
    PREDICT_REGION_DIRS = [
        str(RELEASE_ROOT / "data" / "florida_keys_inversion" / "florida_bay" / "icesat2_selected_predictions"),
        str(RELEASE_ROOT / "data" / "florida_keys_inversion" / "key_largo" / "icesat2_selected_predictions"),
        str(RELEASE_ROOT / "data" / "florida_keys_inversion" / "key_west" / "icesat2_selected_predictions"),
        str(RELEASE_ROOT / "data" / "florida_keys_inversion" / "marathon" / "icesat2_selected_predictions"),
    ]

    # Downsampling strategy:
    # "fps" | "hybrid" | "stride" | "density_fps_stride"
    # density_fps_stride: stage2 density-aware, stage3 FPS, stage4 stride.
    DOWNSAMPLE_STRATEGY = "stride"
    DENSITY_GRID_SIZE = 32
    DENSITY_AWARE_ALPHA = 1.0

    BATCH_SIZE = 4
    EPOCHS = 150
    LR = 1e-4
    WEIGHT_DECAY = 5e-4
    WARMUP_EPOCHS = 10
    SCHEDULER_T0 = 150
    EARLY_STOP_PATIENCE = 10

    SEED = 42
    NUM_WORKERS = 0
