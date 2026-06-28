"""E6: full-scene DEM error analysis.

Outputs per region under Catboost/e6_error_analysis:
  - predicted/cudem/error GeoTIFFs
  - DEM comparison figure
  - depth-bin error CSV
  - region-type summary CSV

Global outputs under DATA_ROOT:
  - e6_depth_bin_summary.csv
  - e6_region_type_summary.csv
  - e6_depth_bin_summary.png
  - e6_region_type_summary.png
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

import config
from utils import compute_metrics, load_prediction_and_aligned_cudem, plot_publication_dem_comparison


DEPTH_BINS = [
    (0, 1, "0-1 m"),
    (1, 2, "1-2 m"),
    (2, 3, "2-3 m"),
    (3, 4, "3-4 m"),
    (4, 5, "4-5 m"),
    (5, 6, "5-6 m"),
    (6, 7, "6-7 m"),
    (7, 8, "7-8 m"),
    (8, 9, "8-9 m"),
    (9, 10, "9-10 m"),
    (10, 11, "10-11 m"),
    (11, 12, "11-12 m"),
    (12,13, "12-13 m"),
    (13,14, "13-14 m"),
    (14, 15, "14-15 m"),
]

REGION_TYPES = {
    "Florida Bay": "Bay",
    "Key Largo": "Reef",
    "Key West": "Nearshore",
    "Marathon": "Nearshore",
}


def predicted_tif(region: str) -> Path:
    safe = config.safe_name(region)
    return config.output_dir(region) / "predictions" / f"{safe}_predicted_depth.tif"


def cudem_path(region: str) -> Path:
    return config.cudem_aligned_path(region)


def load_aligned_arrays(region: str) -> tuple[np.ndarray, np.ndarray, rasterio.Affine, dict]:
    pred_tif = predicted_tif(region)
    if not pred_tif.exists():
        raise FileNotFoundError(f"Missing prediction TIF: {pred_tif}")
    cudem_tif = cudem_path(region)
    if not cudem_tif.exists():
        raise FileNotFoundError(f"Missing CUDEM TIF: {cudem_tif}")
    pred, cudem_depth, profile = load_prediction_and_aligned_cudem(pred_tif, cudem_tif)
    return pred, cudem_depth, profile["transform"], profile


def save_raster(path: Path, template_profile: dict, arr: np.ndarray) -> None:
    profile = template_profile.copy()
    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
    profile["height"], profile["width"] = arr.shape
    out = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def raster_metrics(ref: np.ndarray, pred: np.ndarray) -> dict:
    mask = np.isfinite(ref) & np.isfinite(pred) & (ref >= 0)
    ref = ref[mask]
    pred = pred[mask]
    if len(ref) < 2:
        return {"N": int(len(ref)), "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan}
    return compute_metrics(ref, pred)


def depth_bin_stats(ref: np.ndarray, pred: np.ndarray, region: str) -> list[dict]:
    rows = []
    for lo, hi, label in DEPTH_BINS:
        mask = np.isfinite(ref) & np.isfinite(pred) & (ref >= lo) & (ref < hi)
        r = ref[mask]
        p = pred[mask]
        if len(r) < 2:
            m = {"N": int(len(r)), "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan}
        else:
            m = compute_metrics(r, p)
        rows.append(
            {
                "Region": region,
                "Bin": label,
                "Lower": lo,
                "Upper": hi,
                "N": m["N"],
                "R2": m["R2"],
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "Bias": m["Bias"],
            }
        )
    return rows


def plot_dem_and_error(region: str, pred: np.ndarray, ref: np.ndarray, err: np.ndarray,
                       profile: dict, out_dir: Path) -> None:
    plot_publication_dem_comparison(
        region,
        pred,
        ref,
        err,
        profile,
        out_dir / f"{config.safe_name(region)}_dem_error_summary.png",
    )


def plot_depth_summary(all_rows: list[dict], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"Florida Bay": "#1f77b4", "Key Largo": "#ff7f0e", "Key West": "#2ca02c", "Marathon": "#d62728"}
    x = np.arange(len(DEPTH_BINS))
    labels = [b[2] for b in DEPTH_BINS]
    for region in config.ALL_REGIONS:
        rows = [r for r in all_rows if r["Region"] == region]
        ys = [r["RMSE"] for r in rows]
        ax.plot(x, ys, marker="o", linewidth=2, color=colors.get(region, "gray"), label=region)
    ax.set_xticks(x, labels)
    ax.set_xlabel("CUDEM depth bins")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Depth-binned RMSE by region")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_region_type_summary(rows: list[dict], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(rows))
    rmse = [r["RMSE"] for r in rows]
    mae = [r["MAE"] for r in rows]
    ax.bar(x - 0.18, rmse, width=0.36, label="RMSE")
    ax.bar(x + 0.18, mae, width=0.36, label="MAE")
    ax.set_xticks(x, [r["Region"] for r in rows])
    ax.set_ylabel("Error (m)")
    ax.set_title("Error by region type")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    all_depth_rows = []
    region_type_rows = []

    for region in config.ALL_REGIONS:
        print(f"\n[E6] Region = {region}")
        out_dir = config.output_dir(region) / "e6_error_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)

        pred, ref, _, profile = load_aligned_arrays(region)
        err = pred - ref
        valid = np.isfinite(pred) & np.isfinite(ref) & (ref > 0) & (ref <= 15)
        pred_v = pred[valid]
        ref_v = ref[valid]
        err_v = err[valid]

        save_raster(out_dir / f"{config.safe_name(region)}_predicted_depth.tif", profile, pred)
        save_raster(out_dir / f"{config.safe_name(region)}_cudem_aligned.tif", profile, ref)
        save_raster(out_dir / f"{config.safe_name(region)}_error.tif", profile, err)

        plot_dem_and_error(region, pred, ref, err, profile, out_dir)

        region_metrics = raster_metrics(ref_v, pred_v)
        region_type_rows.append(
            {
                "Region": region,
                "RegionType": REGION_TYPES.get(region, "Other"),
                "ValidPixels": int(valid.sum()),
                "R2": region_metrics["R2"],
                "RMSE": region_metrics["RMSE"],
                "MAE": region_metrics["MAE"],
                "Bias": region_metrics["Bias"],
            }
        )

        depth_rows = depth_bin_stats(ref_v, pred_v, region)
        for row in depth_rows:
            row["RegionType"] = REGION_TYPES.get(region, "Other")
        all_depth_rows.extend(depth_rows)

        with open(out_dir / "depth_bin_metrics.json", "w", encoding="utf-8") as f:
            json.dump(depth_rows, f, indent=2)
        with open(out_dir / "region_type_metrics.json", "w", encoding="utf-8") as f:
            json.dump(region_type_rows[-1], f, indent=2)

    depth_csv = config.DATA_ROOT / "e6_depth_bin_summary.csv"
    with open(depth_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_depth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_depth_rows)

    type_csv = config.DATA_ROOT / "e6_region_type_summary.csv"
    with open(type_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(region_type_rows[0].keys()))
        writer.writeheader()
        writer.writerows(region_type_rows)

    plot_depth_summary(all_depth_rows, config.DATA_ROOT / "e6_depth_bin_summary.png")
    plot_region_type_summary(region_type_rows, config.DATA_ROOT / "e6_region_type_summary.png")

    print(f"[E6] Depth summary CSV: {depth_csv}")
    print(f"[E6] Region-type CSV: {type_csv}")


if __name__ == "__main__":
    main()
