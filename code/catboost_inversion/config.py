"""Configuration for Sentinel-2 CatBoost bathymetry inversion."""
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[2]

CURRENT_REGION = "Florida Bay"
ALL_REGIONS = ["Florida Bay", "Key Largo", "Key West", "Marathon"]
REGION_SLUGS = {
    "Florida Bay": "florida_bay",
    "Key Largo": "key_largo",
    "Key West": "key_west",
    "Marathon": "marathon",
}

FEATURE_COLS = [
    "B02", "B03", "B04", "B08",
    "NDWI", "Blue_Green_LogRatio", "Blue_Red_LogRatio", "Green_Red_LogRatio",
]
TARGET_COL = "Depth"
FEATURE_TIF_NAMES = [f + ".tif" for f in FEATURE_COLS]

PREDICTION_SCENES_BY_REGION = {
    "Florida Bay": "S2B_MSIL2A_20250228T160509_T17RNH_Cropped_CloudMasked",
    "Key Largo": "S2B_MSIL2A_20250320T160509_T17RNH_Cropped_CloudMasked",
    "Key West": "S2B_MSIL2A_20260107T160549_T17RMH_Cropped_CloudMasked",
    "Marathon": "S2B_MSIL2A_20260213T160509_Mosaic_Cropped_CloudMasked",
}

CATBOOST_PARAMS = dict(
    iterations=1000,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=1.0,
    random_strength=0.5,
    bagging_temperature=0.2,
    border_count=254,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,
    thread_count=-1,
    allow_writing_files=False,
)


def safe_name(region: str) -> str:
    return region.replace(" ", "_")


def region_slug(region: str) -> str:
    return REGION_SLUGS[region]


def region_dir(region: str) -> Path:
    return RELEASE_ROOT / "data" / "florida_keys_inversion" / region_slug(region)


def train_dir(region: str) -> Path:
    return region_dir(region) / "matched_samples" / "Train"


def test_dir(region: str) -> Path:
    return region_dir(region) / "matched_samples" / "Test"


def tif_scene_dir(region: str, scene_id: str) -> Path:
    return RELEASE_ROOT / "data" / "sentinel2_selected" / region_slug(region) / "selected_l2a_tif" / scene_id


def tif_scene_dir_final(region: str, scene_id: str) -> Path:
    return tif_scene_dir(region, scene_id)


def prediction_scene_id(region: str) -> str:
    return PREDICTION_SCENES_BY_REGION[region]


def prediction_scene_dir(region: str) -> Path:
    return tif_scene_dir_final(region, prediction_scene_id(region))


def cudem_10m_path(region: str) -> Path:
    return RELEASE_ROOT / "data" / "dem_reference" / region_slug(region) / f"{safe_name(region)}_cudem_10m.tif"


def cudem_aligned_path(region: str) -> Path:
    return cudem_10m_path(region)


def land_mask_path(region: str) -> Path:
    return RELEASE_ROOT / "data" / "dem_reference" / region_slug(region) / f"{safe_name(region)}_land_mask.tif"


def output_dir(region: str) -> Path:
    p = RELEASE_ROOT / "results" / "bathymetry_dem" / region_slug(region)
    p.mkdir(parents=True, exist_ok=True)
    return p
