"""Create an auditable Tutuila MHW-to-MLLW reference and scatter plot.

The existing Tutuila raster was exported from the NOAA DEM_all service and
matches the Tutuila MHW source layer.  The ICESat-2 depth labels are MLLW-
referenced.  At Pago Pago, NOAA reports MHW = 0.807 m relative to MLLW, so
elevation_MLLW = elevation_MHW + 0.807 and depth_MLLW = -elevation_MHW - 0.807.
The original raster is never overwritten; a corrected depth raster, metrics,
and a publication-style PNG are written to a separate audit directory.
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import array_bounds
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(r"H:\ATL24\PhisicKPConvNet")
DATA = ROOT / r"validation_area/data/Six_Regions/07_Tutuila_Pala_Lagoon"
SOURCE = DATA / "Cudem/07_Tutuila_Pala_Lagoon_cudem.tif"
OUT = DATA / "Cudem/tutuila_mllw_correction_audit"
CORRECTED = OUT / "Tutuila_CUDEM_depth_MLLW.tif"
PLOT = OUT / "Tutuila_ICESat2_MLLW_vs_CUDEM_MLLW.png"
METRICS = OUT / "Tutuila_ICESat2_MLLW_vs_CUDEM_MLLW_metrics.json"
POINTS = OUT / "Tutuila_ICESat2_MLLW_vs_CUDEM_MLLW_points.csv"
MHW_MLLW = 0.807
FIGURE05 = ROOT / r"Plot/Manuscript_Figures_Tables/Figure05_ICESat2_Bathymetric_Validation/figure05_bathymetric_validation_self_atl24_combined.py"


def load_figure05():
    spec = importlib.util.spec_from_file_location("figure05_module", FIGURE05)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {FIGURE05}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_corrected_raster() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(SOURCE) as src:
        elevation = src.read(1).astype("float32")
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=-9999.0, compress="DEFLATE")
        depth_mllw = np.full(elevation.shape, -9999.0, dtype="float32")
        valid = np.isfinite(elevation) & (elevation > -9000) & (elevation < 0)
        converted = -elevation[valid] - MHW_MLLW
        keep = converted > 0
        idx = np.flatnonzero(valid)
        depth_mllw.flat[idx[keep]] = converted[keep].astype("float32")
        with rasterio.open(CORRECTED, "w", **profile) as dst:
            dst.write(depth_mllw, 1)
            dst.update_tags(
                reference="CUDEM",
                source_raster=str(SOURCE),
                source_vertical_datum="MHW",
                target_vertical_datum="MLLW",
                conversion="depth_MLLW = -elevation_MHW - 0.807 m",
                mhw_above_mllw_m=MHW_MLLW,
            )


def sample_corrected(rows: list[dict]) -> list[dict]:
    with rasterio.open(CORRECTED) as src:
        coords = [(r["lon"], r["lat"]) for r in rows]
        refs = np.asarray([float(v[0]) for v in src.sample(coords)], dtype=float)
    output = []
    for row, ref in zip(rows, refs):
        if np.isfinite(ref) and ref > 0:
            output.append({"cudem_mllw": float(ref), "icesat2_mllw": float(row["depth"]), "csv": row["csv"]})
    return output


def metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
    return {
        "n": int(len(x)),
        "r2": float(r2_score(x, y)),
        "pearson_r2": corr * corr,
        "mae_m": float(mean_absolute_error(x, y)),
        "rmse_m": float(np.sqrt(mean_squared_error(x, y))),
        "bias_pred_minus_cudem_m": float(np.mean(y - x)),
        "mhw_to_mllw_offset_m": MHW_MLLW,
    }


def density(x: np.ndarray, y: np.ndarray, bins: int = 70) -> np.ndarray:
    upper = max(5.0, math.ceil(float(max(x.max(), y.max())) / 5.0) * 5.0)
    hist, xe, ye = np.histogram2d(x, y, bins=bins, range=[[0, upper], [0, upper]])
    xi = np.clip(np.digitize(x, xe) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, ye) - 1, 0, bins - 1)
    return hist[xi, yi], upper


def make_plot(data: list[dict], stat: dict[str, float]) -> None:
    x = np.asarray([d["cudem_mllw"] for d in data], dtype=float)
    y = np.asarray([d["icesat2_mllw"] for d in data], dtype=float)
    order_density, upper = density(x, y)
    order = np.argsort(order_density)
    max_points = 200_000
    if len(order) > max_points:
        order = order[np.linspace(0, len(order) - 1, max_points).astype(int)]
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.linewidth": 1.0})
    fig, ax = plt.subplots(figsize=(4.4, 4.2), dpi=300)
    sc = ax.scatter(x[order], y[order], c=np.log10(np.maximum(order_density[order], 1)), s=2.0,
                    cmap="turbo", alpha=0.8, linewidths=0, rasterized=True)
    ax.plot([0, upper], [0, upper], "k--", linewidth=1.1)
    ax.set(xlim=(0, upper), ylim=(0, upper), xlabel="CUDEM (m), MLLW", ylabel="ICESat-2 (m), MLLW")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Tutuila", loc="left", fontweight="bold", pad=8)
    # The manuscript-style panel reports squared Pearson correlation, matching
    # the earlier validation figure; the deterministic coefficient of
    # determination remains available in the JSON metrics as ``r2``.
    ax.text(0.56, 0.12, f"N = {stat['n']}\nR² = {stat['pearson_r2']:.2f}\nMAE = {stat['mae_m']:.2f} m\nRMSE = {stat['rmse_m']:.2f} m\nBias = {stat['bias_pred_minus_cudem_m']:+.2f} m",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="log₁₀ point density")
    fig.tight_layout()
    fig.savefig(PLOT, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    make_corrected_raster()
    mod = load_figure05()
    item = next(x for x in mod.REGIONS if x["name"] == "Tutuila")
    tide = mod.FigureTideProvider(mod.TIDE_CACHE, mod.TIDE_PREDICTION_CACHE)
    audit: list[dict] = []
    rows = mod.load_region_rows(item, tide, audit)
    data = sample_corrected(rows)
    x = np.asarray([d["cudem_mllw"] for d in data], dtype=float)
    y = np.asarray([d["icesat2_mllw"] for d in data], dtype=float)
    stat = metrics(x, y)
    POINTS.write_text("cudem_mllw,icesat2_mllw,csv\n" + "\n".join(f"{d['cudem_mllw']:.8f},{d['icesat2_mllw']:.8f},{d['csv']}" for d in data), encoding="utf-8")
    METRICS.write_text(json.dumps({"status": "OK", "region": "Tutuila", "source": str(SOURCE), "corrected_raster": str(CORRECTED), "metrics": stat, "tide_audit_rows": len(audit)}, indent=2, ensure_ascii=False), encoding="utf-8")
    make_plot(data, stat)
    print(json.dumps(stat, indent=2, ensure_ascii=False))
    print(f"corrected_raster={CORRECTED}")
    print(f"plot={PLOT}")


if __name__ == "__main__":
    main()
