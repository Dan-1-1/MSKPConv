"""Predict CatBoost bathymetry and validate against ICESat-2 / CUDEM.

Outputs:
  predictions/{region}_predicted_depth.tif
  predictions/{region}_predicted_depth.png
  predictions/{region}_cudem_scatter.png
  predictions/{region}_icesat2_scatter.png
  predictions/metrics.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from catboost import CatBoostRegressor
from rasterio.warp import Resampling, reproject

import config
from utils import (
    compute_metrics,
    density_scatter,
    load_prediction_and_aligned_cudem,
    plot_publication_depth_map,
)


def load_model(region: str) -> CatBoostRegressor:
    safe = config.safe_name(region)
    path = config.output_dir(region) / f"{safe}_catboost.cbm"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}\nPlease run train.py first.")
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def predict_on_test_csv(model: CatBoostRegressor, region: str) -> pd.DataFrame:
    """Predict on matched ICESat-2/Sentinel-2 test-point CSV files."""
    files = sorted(config.test_dir(region).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No test CSV in {config.test_dir(region)}")

    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    valid = df[config.FEATURE_COLS].notna().all(axis=1)
    df = df.loc[valid].reset_index(drop=True)
    x = df[config.FEATURE_COLS].astype("float32").values
    df["Depth_Pred"] = model.predict(x).astype("float32")
    return df


def predict_full_scene_tif(model: CatBoostRegressor, region: str,
                           scene_id: str, out_tif: Path) -> Path | None:
    """Predict all valid pixels in the selected Sentinel-2 feature TIF stack."""
    scene_dir = config.tif_scene_dir_final(region, scene_id)
    if not scene_dir.exists():
        print(f"[WARN] Scene TIF folder not found: {scene_dir}; skip TIF prediction")
        return None

    arrays = []
    profile = None
    for fname in config.FEATURE_TIF_NAMES:
        path = scene_dir / fname
        if not path.exists():
            print(f"[WARN] missing {path}; skip TIF prediction")
            return None
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

    # Apply land mask: set land pixels to NoData
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
            # mask == 1 is land, mask == 0 is water, 255 is NoData
            pred[(mask == 1) | (mask == 255)] = np.nan
        print(f"[INFO] Land mask applied: {land_mask_path.name}")
    else:
        print(f"[WARN] Land mask not found: {land_mask_path}; skipping land masking")

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
    profile.setdefault("height", pred.shape[0])
    profile.setdefault("width", pred.shape[1])
    out_arr = np.where(np.isfinite(pred), pred, -9999.0).astype("float32")
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(out_arr, 1)
    print(f"Saved depth TIF: {out_tif}")

    out_png = out_tif.with_suffix(".png")
    plot_publication_depth_map(
        pred,
        profile,
        out_png,
        title=f"{region} - CatBoost predicted bathymetry",
        subtitle=f"Scene: {scene_id}",
    )
    print(f"Saved depth visualization: {out_png}")
    return out_tif


def cudem_validation(pred_tif: Path, region: str, out_png: Path,
                     title_suffix: str = "") -> dict:
    """Validate full-scene prediction raster against pre-aligned CUDEM."""
    pred, cudem_depth, _ = load_prediction_and_aligned_cudem(
        pred_tif,
        config.cudem_aligned_path(region),
    )
    mask = (
        np.isfinite(cudem_depth)
        & np.isfinite(pred)
        & (cudem_depth > 0)
        & (cudem_depth <= 15)
    )
    cudem_values = cudem_depth[mask]
    pred_values = pred[mask]
    metrics = compute_metrics(cudem_values, pred_values)

    title = f"{region} - CatBoost vs CUDEM" + (f"\n{title_suffix}" if title_suffix else "")
    density_scatter(
        cudem_values,
        pred_values,
        metrics,
        out_png,
        title=title,
        xlabel="CUDEM Depth (m)",
        ylabel="CatBoost Predicted Depth (m)",
    )
    return metrics


def icesat2_validation(df: pd.DataFrame, region: str, out_png: Path,
                       title_suffix: str = "") -> dict:
    y_true = df[config.TARGET_COL].values
    y_pred = df["Depth_Pred"].values
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_true <= 15)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    metrics = compute_metrics(y_true, y_pred)
    title = f"{region} - CatBoost vs ICESat-2" + (f"\n{title_suffix}" if title_suffix else "")
    density_scatter(
        y_true,
        y_pred,
        metrics,
        out_png,
        title=title,
        xlabel="ICESat-2 Depth (m)",
        ylabel="CatBoost Predicted Depth (m)",
    )
    return metrics


def run(region: str | None = None) -> None:
    region = region or config.CURRENT_REGION
    print(f"\n{'=' * 60}\n[Predict + Validate] Region = {region}\n{'=' * 60}")

    model = load_model(region)
    df = predict_on_test_csv(model, region)
    print(f"Predicted {len(df):,} matched test points")

    out = config.output_dir(region) / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    safe = config.safe_name(region)

    ic_png = out / f"{safe}_icesat2_scatter.png"
    ic_metrics = icesat2_validation(df, region, ic_png)
    print(
        f"[ICESat-2 matched points] R2={ic_metrics['R2']:.3f} | "
        f"RMSE={ic_metrics['RMSE']:.3f} | MAE={ic_metrics['MAE']:.3f} | "
        f"Bias={ic_metrics['Bias']:+.3f} | N={ic_metrics['N']}"
    )

    scene_id = config.prediction_scene_id(region)
    tif_path = out / f"{safe}_predicted_depth.tif"
    pred_tif = predict_full_scene_tif(model, region, scene_id, tif_path)

    cu_png = out / f"{safe}_cudem_scatter.png"
    cu_metrics = cudem_validation(pred_tif, region, cu_png)
    print(
        f"[CUDEM full scene] R2={cu_metrics['R2']:.3f} | "
        f"RMSE={cu_metrics['RMSE']:.3f} | MAE={cu_metrics['MAE']:.3f} | "
        f"Bias={cu_metrics['Bias']:+.3f} | N={cu_metrics['N']}"
    )

    all_metrics = {
        "region": region,
        "scene_id": str(scene_id),
        "icesat2_test": ic_metrics,
        "cudem_validation": cu_metrics,
        "cudem_validation_source": "full_scene_prediction_tif",
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for region_name in config.ALL_REGIONS:
            run(region_name)
    else:
        run(args.region)
