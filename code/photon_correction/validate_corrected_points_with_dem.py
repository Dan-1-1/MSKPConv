import argparse
import math
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.stats import gaussian_kde


BASE_DIR = Path(r"H:\ATL24\PhisicKPConvNet\validation_area\data")

REGIONS = {
    "Florida Bay": {
        "root": BASE_DIR / "Florida Bay",
        "cudem": BASE_DIR / "Florida Bay" / "Cudem" / "Florida_Bay_cudem.tif",
    },
    "Key West": {
        "root": BASE_DIR / "Key West",
        "cudem": BASE_DIR / "Key West" / "Cudem" / "Key_West_cudem.tif",
    },
    "Marathon": {
        "root": BASE_DIR / "Marathon",
        "cudem": BASE_DIR / "Marathon" / "Cudem" / "Marathon_cudem.tif",
    },
    "Key Largo": {
        "root": BASE_DIR / "Key Largo",
        "cudem": BASE_DIR / "Key Largo" / "Cudem" / "Key_Largo_cudem.tif",
    },
}


def parse_title(csv_path: Path) -> str:
    match = re.search(r"ATL03_(\d{8})\d{6}.*_(gt[123][lr])(?:_|$)", csv_path.stem, re.I)
    if match:
        return f"{match.group(1)}{match.group(2).upper()}"
    return csv_path.stem


def to_positive_depth(values: np.ndarray) -> np.ndarray:
    arr = values.astype("float64", copy=True)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return arr
    sample = arr[valid]
    negative_ratio = np.mean(sample < 0)
    positive_ratio = np.mean(sample > 0)
    if negative_ratio > positive_ratio:
        arr = -arr
    return arr


def format_r2(r2: float) -> str:
    if not np.isfinite(r2):
        return "nan"
    return f"{r2:.2f}"


def compute_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - reference
    mae = np.mean(np.abs(error))
    rmse = math.sqrt(np.mean(error * error))
    if len(reference) > 1 and np.std(reference) > 0 and np.std(predicted) > 0:
        r = np.corrcoef(reference, predicted)[0, 1]
        r2 = r * r
    else:
        r2 = np.nan
    return {"n": len(reference), "r2": r2, "mae": mae, "rmse": rmse}


def density_colors(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return np.ones_like(x)
    try:
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
    except Exception:
        z = np.ones_like(x)
    return z


def plot_correlation(reference: np.ndarray, predicted: np.ndarray, title: str, out_png: Path) -> None:
    metrics = compute_metrics(reference, predicted)
    z = density_colors(reference, predicted)
    order = np.argsort(z)
    x = reference[order]
    y = predicted[order]
    z = z[order]

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=300)
    ax.scatter(
        x,
        y,
        c=z,
        s=9,
        cmap="jet",
        alpha=0.9,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot([0, 30], [0, 30], color="red", linewidth=1.5)

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("CUDEM (m)", fontsize=11)
    ax.set_ylabel("ICESat-2 (m)", fontsize=11)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_yticks(np.arange(0, 31, 5))
    ax.tick_params(length=3.5, width=1.0, pad=2, labelsize=9)

    ax.text(
        0.51,
        0.95,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    stats_text = (
        f"N = {metrics['n']}\n"
        f"R$^2$ = {format_r2(metrics['r2'])}\n"
        f"MAE = {metrics['mae']:.2f} m\n"
        f"RMSE = {metrics['rmse']:.2f} m"
    )
    ax.text(
        0.60,
        0.13,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def collect_matches(csv_path: Path, cudem_path: Path, chunksize: int) -> tuple[np.ndarray, np.ndarray]:
    ref_parts = []
    pred_parts = []
    usecols = ["Latitude", "Longitude", "Depth"]

    with rasterio.open(cudem_path) as src:
        transformer = None
        if src.crs and str(src.crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

        nodata = src.nodata
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
            lon = pd.to_numeric(chunk["Longitude"], errors="coerce").to_numpy("float64")
            lat = pd.to_numeric(chunk["Latitude"], errors="coerce").to_numpy("float64")
            depth = pd.to_numeric(chunk["Depth"], errors="coerce").to_numpy("float64")

            valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(depth)
            if not np.any(valid):
                continue

            lon = lon[valid]
            lat = lat[valid]
            depth = depth[valid]

            if transformer is not None:
                xs, ys = transformer.transform(lon, lat)
            else:
                xs, ys = lon, lat

            coords = list(zip(xs, ys))
            sampled = np.fromiter((v[0] for v in src.sample(coords)), dtype="float64", count=len(coords))
            ok = np.isfinite(sampled) & np.isfinite(depth)
            if nodata is not None and np.isfinite(nodata):
                ok &= sampled != nodata
            if not np.any(ok):
                continue

            ref_parts.append(sampled[ok])
            pred_parts.append(depth[ok])

    if not ref_parts:
        return np.array([], dtype="float64"), np.array([], dtype="float64")

    reference = to_positive_depth(np.concatenate(ref_parts))
    predicted = to_positive_depth(np.concatenate(pred_parts))
    valid_depth = (
        np.isfinite(reference)
        & np.isfinite(predicted)
        & (reference >= 0)
        & (reference <= 30)
        & (predicted >= 0)
        & (predicted <= 30)
    )
    return reference[valid_depth], predicted[valid_depth]


def safe_output_name(csv_path: Path) -> str:
    return f"{parse_title(csv_path)}_correlation.png"


def iter_prediction_csvs(pred_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in pred_dir.glob("*.csv")
        if p.name.lower() != "file_metrics.csv" and p.name.startswith("ATL03_")
    )


def copy_selected_files(csv_path: Path, correlation_png: Path, selected_dir: Path) -> list[str]:
    selected_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    result_png = csv_path.with_name(f"{csv_path.stem}_result.png")
    targets = [
        (csv_path, selected_dir / csv_path.name),
        (result_png, selected_dir / result_png.name),
        (correlation_png, selected_dir / correlation_png.name),
    ]

    for source, target in targets:
        if source.exists():
            shutil.copy2(source, target)
            copied.append(source.name)
        else:
            print(f"[warn] missing selected companion file: {source}")

    return copied


def select_region_by_r2(
    region_name: str,
    cfg: dict,
    chunksize: int,
    threshold: float,
    max_files: int | None,
) -> tuple[int, int]:
    root = cfg["root"]
    pred_dir = root / "Predictions"
    cudem_path = cfg["cudem"]
    corr_dir = root / "ICESat2_CUDEM_Correlation"
    selected_dir = root / "Predictions_selected"

    if not cudem_path.exists():
        raise FileNotFoundError(f"CUDEM not found for {region_name}: {cudem_path}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found for {region_name}: {pred_dir}")
    if not corr_dir.exists():
        raise FileNotFoundError(f"Correlation directory not found for {region_name}: {corr_dir}")

    selected = 0
    skipped = 0
    csv_paths = iter_prediction_csvs(pred_dir)
    if max_files is not None:
        csv_paths = csv_paths[:max_files]
    print(f"[{region_name}] selecting R2 > {threshold} from {len(csv_paths)} CSV files")

    for index, csv_path in enumerate(csv_paths, start=1):
        corr_png = corr_dir / safe_output_name(csv_path)
        if not corr_png.exists():
            print(f"[{region_name}] skip missing correlation png ({index}/{len(csv_paths)}): {corr_png.name}")
            skipped += 1
            continue

        try:
            reference, predicted = collect_matches(csv_path, cudem_path, chunksize)
            if len(reference) < 3:
                print(f"[{region_name}] skip no-data ({index}/{len(csv_paths)}): {csv_path.name}, N={len(reference)}")
                skipped += 1
                continue

            metrics = compute_metrics(reference, predicted)
            if np.isfinite(metrics["r2"]) and metrics["r2"] > threshold:
                copied = copy_selected_files(csv_path, corr_png, selected_dir)
                selected += 1
                print(
                    f"[{region_name}] selected ({index}/{len(csv_paths)}): "
                    f"{csv_path.name}, R2={metrics['r2']:.3f}, copied={len(copied)}"
                )
            else:
                skipped += 1
        except Exception as exc:
            skipped += 1
            print(f"[{region_name}] failed selection ({index}/{len(csv_paths)}): {csv_path.name}: {exc}")

    return selected, skipped


def process_region(region_name: str, cfg: dict, chunksize: int, overwrite: bool, max_files: int | None) -> tuple[int, int]:
    root = cfg["root"]
    pred_dir = root / "Predictions"
    cudem_path = cfg["cudem"]
    out_dir = root / "ICESat2_CUDEM_Correlation"

    if not cudem_path.exists():
        raise FileNotFoundError(f"CUDEM not found for {region_name}: {cudem_path}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found for {region_name}: {pred_dir}")

    csv_paths = iter_prediction_csvs(pred_dir)
    if max_files is not None:
        csv_paths = csv_paths[:max_files]
    made = 0
    skipped = 0
    print(f"[{region_name}] {len(csv_paths)} CSV files")
    print(f"[{region_name}] CUDEM: {cudem_path}")

    for index, csv_path in enumerate(csv_paths, start=1):
        out_png = out_dir / safe_output_name(csv_path)
        if out_png.exists() and not overwrite:
            skipped += 1
            continue

        try:
            reference, predicted = collect_matches(csv_path, cudem_path, chunksize)
            if len(reference) < 3:
                print(f"[{region_name}] skip no-data ({index}/{len(csv_paths)}): {csv_path.name}, N={len(reference)}")
                skipped += 1
                continue
            plot_correlation(reference, predicted, parse_title(csv_path), out_png)
            made += 1
            print(f"[{region_name}] saved ({index}/{len(csv_paths)}): {out_png.name}, N={len(reference)}")
        except Exception as exc:
            skipped += 1
            print(f"[{region_name}] failed ({index}/{len(csv_paths)}): {csv_path.name}: {exc}")

    return made, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create ICESat-2 vs CUDEM 0-30 m correlation plots for validation regions."
    )
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV rows read per chunk.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing PNG files.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional per-region file limit for testing.")
    parser.add_argument("--copy-selected", action="store_true", help="Copy selected Florida Bay CSV/PNG files by R2.")
    parser.add_argument("--select-r2-threshold", type=float, default=0.9, help="R2 threshold for selected copies.")
    parser.add_argument(
        "--region",
        action="append",
        choices=sorted(REGIONS),
        help="Optional region name. Can be supplied multiple times.",
    )
    args = parser.parse_args()

    if args.copy_selected:
        region_names = args.region or list(REGIONS)
        total_selected = 0
        total_skipped = 0
        for region_name in region_names:
            selected, skipped = select_region_by_r2(
                region_name,
                REGIONS[region_name],
                args.chunksize,
                args.select_r2_threshold,
                args.max_files,
            )
            total_selected += selected
            total_skipped += skipped
        print(f"Done. Selected {total_selected} CSV groups, skipped {total_skipped} files.")
        return

    region_names = args.region or list(REGIONS)
    total_made = 0
    total_skipped = 0
    for region_name in region_names:
        made, skipped = process_region(
            region_name,
            REGIONS[region_name],
            args.chunksize,
            args.overwrite,
            args.max_files,
        )
        total_made += made
        total_skipped += skipped
    print(f"Done. Created {total_made} plots, skipped {total_skipped} files.")


if __name__ == "__main__":
    main()
