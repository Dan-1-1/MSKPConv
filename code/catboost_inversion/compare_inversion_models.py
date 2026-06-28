"""Compare machine-learning and empirical SDB models for Sentinel-2 bathymetry.

Models:
  1. Random Forest
  2. XGBoost
  3. Lyzenga linear log-band model
  4. Stumpf blue/green log-ratio model
  5. CatBoost

The script reuses config.py and utils.py in the CatBoost folder. It trains each
model on the matched Sentinel-2/ICESat-2 training CSVs, evaluates independent
test CSVs, predicts the configured full Sentinel-2 scene, and validates the
full-scene DEM against the aligned CUDEM reference.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from catboost import CatBoostRegressor
from rasterio.warp import Resampling, reproject
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import config
from utils import (
    compute_metrics,
    density_scatter,
    load_prediction_and_aligned_cudem,
    load_region_csvs,
    plot_publication_depth_map,
)


MODEL_ORDER = ["random_forest", "xgboost", "lyzenga", "stumpf_log_ratio", "catboost"]
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lyzenga": "Lyzenga",
    "stumpf_log_ratio": "Stumpf log-ratio",
    "catboost": "CatBoost",
}


class LyzengaRegressor:
    """Linear SDB model using log-transformed visible bands."""

    band_indices = (0, 1, 2)  # B02, B03, B04 in config.FEATURE_COLS

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.model = LinearRegression()

    def _transform(self, x) -> np.ndarray:
        x = np.asarray(x, dtype="float64")
        bands = np.maximum(x[:, self.band_indices], self.eps)
        return np.log(bands)

    def fit(self, x, y):
        self.model.fit(self._transform(x), y)
        return self

    def predict(self, x) -> np.ndarray:
        return self.model.predict(self._transform(x))


class StumpfLogRatioRegressor:
    """Linear SDB model using the blue/green log-ratio feature."""

    blue_idx = 0
    green_idx = 1

    def __init__(self, scale: float = 1000.0, eps: float = 1e-6):
        self.scale = scale
        self.eps = eps
        self.model = LinearRegression()

    def _transform(self, x) -> np.ndarray:
        x = np.asarray(x, dtype="float64")
        blue = np.maximum(x[:, self.blue_idx], self.eps)
        green = np.maximum(x[:, self.green_idx], self.eps)
        ratio = np.log(self.scale * blue + self.eps) / np.log(self.scale * green + self.eps)
        return ratio.reshape(-1, 1)

    def fit(self, x, y):
        self.model.fit(self._transform(x), y)
        return self

    def predict(self, x) -> np.ndarray:
        return self.model.predict(self._transform(x))


def build_model(model_name: str):
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            max_features="sqrt",
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "xgboost":
        return XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
    if model_name == "lyzenga":
        return LyzengaRegressor()
    if model_name == "stumpf_log_ratio":
        return StumpfLogRatioRegressor()
    if model_name == "catboost":
        params = dict(config.CATBOOST_PARAMS)
        params["verbose"] = False
        return CatBoostRegressor(**params)
    raise ValueError(f"Unknown model: {model_name}")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=config.FEATURE_COLS + [config.TARGET_COL]).copy()
    valid = np.isfinite(df[config.FEATURE_COLS + [config.TARGET_COL]].astype("float64")).all(axis=1)
    return df.loc[valid].reset_index(drop=True)


def load_region_csvs_limited(folder: Path, max_files: int | None = None) -> pd.DataFrame:
    """Read CSV files from a folder, optionally limiting the number of files."""
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))
    if max_files is not None:
        if max_files <= 0:
            raise ValueError("--max-train-files must be a positive integer")
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    dfs = [pd.read_csv(f, encoding="utf-8-sig") for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_train_test(region: str, max_train_files: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = clean_dataframe(load_region_csvs_limited(config.train_dir(region), max_train_files))
    test_df = clean_dataframe(load_region_csvs(config.test_dir(region)))
    return train_df, test_df


def xy_from_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df[config.FEATURE_COLS].astype("float32").values
    y = df[config.TARGET_COL].astype("float32").values
    return x, y


def model_dir(region: str, model_name: str) -> Path:
    return config.output_dir(region) / "multi_model_comparison" / model_name


def save_model(model, model_name: str, out_dir: Path, safe_region: str) -> Path:
    if model_name == "catboost":
        path = out_dir / f"{safe_region}_{model_name}.cbm"
        model.save_model(str(path))
    else:
        path = out_dir / f"{safe_region}_{model_name}.joblib"
        joblib.dump(model, path)
    return path


def predict_test_points(model, test_df: pd.DataFrame) -> pd.DataFrame:
    x, _ = xy_from_df(test_df)
    out = test_df.copy()
    out["Depth_Pred"] = model.predict(x).astype("float32")
    return out


def save_raster(path: Path, profile: dict, arr: np.ndarray) -> None:
    profile = profile.copy()
    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
    profile["height"], profile["width"] = arr.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    out_arr = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out_arr, 1)


def predict_full_scene_tif(model, region: str, model_name: str, out_tif: Path) -> Path:
    scene_id = config.prediction_scene_id(region)
    scene_dir = config.prediction_scene_dir(region)
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene TIF folder not found: {scene_dir}")

    arrays = []
    profile = None
    for fname in config.FEATURE_TIF_NAMES:
        path = scene_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing feature TIF: {path}")
        with rasterio.open(path) as src:
            arrays.append(src.read(1).astype("float32"))
            if profile is None:
                profile = src.profile.copy()

    stack = np.stack(arrays, axis=-1)
    valid = np.all(np.isfinite(stack), axis=-1)
    valid = valid & np.all(stack[..., :4] > 0, axis=-1)

    pred = np.full(stack.shape[:2], np.nan, dtype="float32")
    if np.any(valid):
        pred[valid] = model.predict(stack[valid]).astype("float32")

    land_mask_path = config.land_mask_path(region)
    if land_mask_path.exists():
        with rasterio.open(land_mask_path) as lm_src:
            if (
                lm_src.shape == pred.shape
                and lm_src.transform == profile["transform"]
                and str(lm_src.crs) == str(profile.get("crs"))
            ):
                mask = lm_src.read(1)
            else:
                mask = np.empty(pred.shape, dtype="uint8")
                reproject(
                    source=rasterio.band(lm_src, 1),
                    destination=mask,
                    src_transform=lm_src.transform,
                    src_crs=lm_src.crs,
                    dst_transform=profile["transform"],
                    dst_crs=profile.get("crs"),
                    dst_nodata=255,
                    resampling=Resampling.nearest,
                )
            pred[(mask == 1) | (mask == 255)] = np.nan

    save_raster(out_tif, profile, pred)
    plot_publication_depth_map(
        pred,
        profile,
        out_tif.with_suffix(".png"),
        title=f"{region} - {MODEL_LABELS[model_name]} predicted bathymetry",
        subtitle=f"Scene: {scene_id}",
    )
    return out_tif


def validate_test_points(df: pd.DataFrame, region: str, model_name: str, out_png: Path) -> dict:
    y_true = df[config.TARGET_COL].values
    y_pred = df["Depth_Pred"].values
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_true <= 15)
    metrics = compute_metrics(y_true[mask], y_pred[mask])
    density_scatter(
        y_true[mask],
        y_pred[mask],
        metrics,
        out_png,
        title=f"{region} - {MODEL_LABELS[model_name]} vs ICESat-2",
        xlabel="ICESat-2 Depth (m)",
        ylabel=f"{MODEL_LABELS[model_name]} Predicted Depth (m)",
    )
    return metrics


def validate_cudem(pred_tif: Path, region: str, model_name: str, out_png: Path) -> dict:
    pred, cudem_depth, _ = load_prediction_and_aligned_cudem(pred_tif, config.cudem_aligned_path(region))
    mask = np.isfinite(pred) & np.isfinite(cudem_depth) & (cudem_depth > 0) & (cudem_depth <= 15)
    metrics = compute_metrics(cudem_depth[mask], pred[mask])
    density_scatter(
        cudem_depth[mask],
        pred[mask],
        metrics,
        out_png,
        title=f"{region} - {MODEL_LABELS[model_name]} vs CUDEM",
        xlabel="CUDEM Depth (m)",
        ylabel=f"{MODEL_LABELS[model_name]} Predicted Depth (m)",
    )
    return metrics


def run_one_model(
    region: str,
    model_name: str,
    skip_full_scene: bool = False,
    max_train_files: int | None = None,
) -> dict:
    print(f"\n[Multi-model] Region={region} | Model={MODEL_LABELS[model_name]}")
    if max_train_files is not None:
        print(f"  Using first {max_train_files} training CSV files")
    safe = config.safe_name(region)
    out_dir = model_dir(region, model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_train_test(region, max_train_files=max_train_files)
    x, y = xy_from_df(train_df)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, random_state=42)

    model = build_model(model_name)
    if model_name == "catboost":
        model.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True)
    else:
        model.fit(x_train, y_train)

    valid_pred = model.predict(x_valid)
    valid_metrics = compute_metrics(y_valid, valid_pred)
    test_pred_df = predict_test_points(model, test_df)
    test_metrics = validate_test_points(
        test_pred_df,
        region,
        model_name,
        out_dir / f"{safe}_{model_name}_icesat2_scatter.png",
    )
    test_pred_df.to_csv(out_dir / f"{safe}_{model_name}_test_predictions.csv", index=False, encoding="utf-8-sig")

    model_path = save_model(model, model_name, out_dir, safe)
    pred_tif = None
    cudem_metrics = {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "N": 0}
    if not skip_full_scene:
        pred_tif = predict_full_scene_tif(
            model,
            region,
            model_name,
            out_dir / f"{safe}_{model_name}_predicted_depth.tif",
        )
        cudem_metrics = validate_cudem(
            pred_tif,
            region,
            model_name,
            out_dir / f"{safe}_{model_name}_cudem_scatter.png",
        )

    result = {
        "Region": region,
        "Model": MODEL_LABELS[model_name],
        "ModelKey": model_name,
        "MaxTrainFiles": int(max_train_files) if max_train_files is not None else "",
        "N_train": int(len(train_df)),
        "N_test": int(len(test_pred_df)),
        "TrainValidation_R2": valid_metrics["R2"],
        "TrainValidation_RMSE": valid_metrics["RMSE"],
        "TrainValidation_MAE": valid_metrics["MAE"],
        "TrainValidation_Bias": valid_metrics["Bias"],
        "ICESat2_R2": test_metrics["R2"],
        "ICESat2_RMSE": test_metrics["RMSE"],
        "ICESat2_MAE": test_metrics["MAE"],
        "ICESat2_Bias": test_metrics["Bias"],
        "ICESat2_N": test_metrics["N"],
        "CUDEM_R2": cudem_metrics["R2"],
        "CUDEM_RMSE": cudem_metrics["RMSE"],
        "CUDEM_MAE": cudem_metrics["MAE"],
        "CUDEM_Bias": cudem_metrics["Bias"],
        "CUDEM_N": cudem_metrics["N"],
        "ModelPath": str(model_path),
        "PredictionTif": str(pred_tif) if pred_tif is not None else "",
        "OutputDir": str(out_dir),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f"  ICESat-2 RMSE={test_metrics['RMSE']:.3f}, MAE={test_metrics['MAE']:.3f}, "
        f"R2={test_metrics['R2']:.3f}"
    )
    if not skip_full_scene:
        print(
            f"  CUDEM    RMSE={cudem_metrics['RMSE']:.3f}, MAE={cudem_metrics['MAE']:.3f}, "
            f"R2={cudem_metrics['R2']:.3f}"
        )

    del model, train_df, test_df, test_pred_df
    gc.collect()
    return result


def write_summary(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict], out_png: Path) -> None:
    df = pd.DataFrame(rows)
    regions = list(config.ALL_REGIONS)
    models = [MODEL_LABELS[m] for m in MODEL_ORDER if MODEL_LABELS[m] in set(df["Model"])]
    colors = {
        "Random Forest": "#4C78A8",
        "XGBoost": "#F58518",
        "Lyzenga": "#54A24B",
        "Stumpf log-ratio": "#B279A2",
        "CatBoost": "#E45756",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "ICESat2_RMSE", "ICESat-2 test RMSE"),
        (axes[1], "CUDEM_RMSE", "CUDEM consistency RMSE"),
    ]:
        x = np.arange(len(regions))
        width = 0.78 / max(len(models), 1)
        for i, model_label in enumerate(models):
            values = []
            for region in regions:
                subset = df[(df["Region"] == region) & (df["Model"] == model_label)]
                values.append(float(subset.iloc[0][metric]) if len(subset) else np.nan)
            ax.bar(
                x - 0.39 + width / 2 + i * width,
                values,
                width=width,
                label=model_label,
                color=colors.get(model_label),
            )
        ax.set_xticks(x, regions, rotation=15, ha="right")
        ax.set_ylabel("RMSE (m)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


def parse_models(models_arg: str) -> list[str]:
    if models_arg.lower() == "all":
        return MODEL_ORDER
    requested = [m.strip().lower() for m in models_arg.split(",") if m.strip()]
    unknown = [m for m in requested if m not in MODEL_ORDER]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Valid models: {MODEL_ORDER}")
    return requested


def selected_regions(region: str | None, run_all: bool) -> Iterable[str]:
    if run_all:
        return config.ALL_REGIONS
    return [region or config.CURRENT_REGION]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=None, help="Region name; defaults to config.CURRENT_REGION")
    parser.add_argument("--all", action="store_true", help="Run all configured regions")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys or 'all'. Keys: random_forest,xgboost,lyzenga,stumpf_log_ratio,catboost",
    )
    parser.add_argument("--skip-full-scene", action="store_true", help="Skip full-scene TIF prediction and CUDEM validation")
    parser.add_argument(
        "--max-train-files",
        type=int,
        default=None,
        help="Use only the first N training CSV files per region; default uses all training CSV files",
    )
    args = parser.parse_args()

    models = parse_models(args.models)
    rows = []
    for region in selected_regions(args.region, args.all):
        for model_name in models:
            rows.append(
                run_one_model(
                    region,
                    model_name,
                    skip_full_scene=args.skip_full_scene,
                    max_train_files=args.max_train_files,
                )
            )

    summary_dir = config.DATA_ROOT / "multi_model_comparison"
    summary_csv = summary_dir / "multi_model_comparison_summary.csv"
    write_summary(rows, summary_csv)
    plot_summary(rows, summary_dir / "multi_model_comparison_rmse_summary.png")
    print(f"\n[Multi-model] Summary CSV: {summary_csv}")
    print(f"[Multi-model] Summary plot: {summary_dir / 'multi_model_comparison_rmse_summary.png'}")


if __name__ == "__main__":
    main()
