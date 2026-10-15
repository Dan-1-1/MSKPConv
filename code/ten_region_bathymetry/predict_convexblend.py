"""Run the released five-model ConvexBlend on aligned Sentinel-2 samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "B02",
    "B03",
    "B04",
    "B08",
    "NDWI",
    "Blue_Green_LogRatio",
    "Blue_Red_LogRatio",
    "Green_Red_LogRatio",
]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("--input must be a .csv or .parquet file")


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError("--output must be a .csv or .parquet file")


def _model_path(model_root: Path, name: str) -> Path:
    suffix = ".cbm" if name == "CatBoost" else ".joblib"
    return model_root / name / f"final{suffix}"


def _load_blend(model_root: Path) -> tuple[list[str], np.ndarray]:
    metadata_path = model_root / "ConvexBlend" / "final.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    names = [str(value) for value in metadata["base_regressors"]]
    weights = np.asarray(metadata["weights"], dtype="float64")
    if len(names) != len(weights) or len(names) != 5:
        raise ValueError("ConvexBlend metadata must describe exactly five models")
    if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise ValueError("ConvexBlend weights must be non-negative and sum to one")
    missing = [str(_model_path(model_root, name)) for name in names if not _model_path(model_root, name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing released base checkpoint(s):\n  "
            + "\n  ".join(missing)
            + "\nRun: python scripts/download_convexblend_weights.py"
        )
    return names, weights


def predict(input_path: Path, patches_path: Path, output_path: Path, model_root: Path) -> None:
    # The pickled CNN class retains its training-time module name.
    package_root = Path(__file__).resolve().parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    from catboost import CatBoostRegressor
    from joblib import load

    frame = _read_table(input_path)
    missing_columns = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")
    feature_frame = frame[FEATURE_COLUMNS].astype("float64")
    if not np.isfinite(feature_frame.to_numpy()).all():
        raise ValueError("Input feature table contains non-finite values")

    patches = np.load(patches_path, mmap_mode="r")
    if patches.shape[0] != len(frame) or patches.ndim != 4 or patches.shape[1] != 4:
        raise ValueError(
            f"CNN patches must have shape [N,4,H,W] and align with input rows; got {patches.shape}"
        )

    names, weights = _load_blend(model_root)
    base_predictions: list[np.ndarray] = []
    result = frame.copy()
    for name in names:
        path = _model_path(model_root, name)
        if name == "CatBoost":
            model = CatBoostRegressor()
            model.load_model(str(path))
        else:
            model = load(path)
        values = patches if name == "LightweightCNN" else feature_frame
        prediction = np.asarray(model.predict(values), dtype="float64").reshape(-1)
        if len(prediction) != len(frame) or not np.isfinite(prediction).all():
            raise ValueError(f"{name} returned incomplete or non-finite predictions")
        result[f"Prediction_{name}"] = prediction
        base_predictions.append(prediction)

    matrix = np.column_stack(base_predictions)
    result["Depth_Prediction_ConvexBlend"] = matrix @ weights
    _write_table(result, output_path)


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV or Parquet feature table")
    parser.add_argument("--patches", type=Path, required=True, help="Aligned [N,4,H,W] NumPy array")
    parser.add_argument("--output", type=Path, required=True, help="CSV or Parquet prediction table")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=repository_root / "models" / "convexblend" / "final" / "Proposed",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predict(args.input, args.patches, args.output, args.model_root)
    print(f"Wrote ConvexBlend predictions to {args.output}")


if __name__ == "__main__":
    main()
