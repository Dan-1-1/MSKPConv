"""Shared utilities for data loading, metrics, raster sampling, and plots."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from sklearn.metrics import mean_absolute_error, mean_squared_error


BATHYMETRY_CMAP = LinearSegmentedColormap.from_list(
    "publication_bathymetry",
    ["#f7fbff", "#d6ecf4", "#9ecae1", "#4292c6", "#08519c", "#08306b"],
)
DEPTH_VMIN = 0.0
DEPTH_VMAX = 15.0
ERROR_VMAX = 3.0


def load_region_csvs(folder: Path) -> pd.DataFrame:
    """Read all CSV files in a folder and concatenate them into one DataFrame."""
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    dfs = [pd.read_csv(f, encoding="utf-8-sig") for f in files]
    return pd.concat(dfs, ignore_index=True)


def compute_metrics(y_true, y_pred) -> dict:
    """Compute squared Pearson correlation, RMSE, MAE, Bias, and N."""
    y_true = np.asarray(y_true).ravel().astype("float64")
    y_pred = np.asarray(y_pred).ravel().astype("float64")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {
            "R2": float("nan"),
            "RMSE": float("nan"),
            "MAE": float("nan"),
            "Bias": float("nan"),
            "N": int(mask.sum()),
        }

    yt, yp = y_true[mask], y_pred[mask]
    if np.std(yt) > 0 and np.std(yp) > 0:
        r = np.corrcoef(yt, yp)[0, 1]
        r2 = r * r
    else:
        r2 = np.nan

    return {
        "R2": float(r2),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
        "Bias": float(np.mean(yp - yt)),
        "N": int(len(yt)),
    }


def _clean_nodata(arr: np.ndarray, nodata) -> np.ndarray:
    arr = arr.astype("float64")
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where(np.isclose(arr, -9999.0), np.nan, arr)
    return arr


def _read_overlap(src, bounds, transform) -> np.ndarray:
    window = from_bounds(*bounds, transform=transform)
    window = window.round_offsets().round_lengths()
    return src.read(1, window=window).astype("float64")


def load_prediction_and_aligned_cudem(pred_tif: Path, cudem_tif: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a prediction raster and a pre-aligned CUDEM raster without resampling."""
    pred_tif = Path(pred_tif)
    cudem_tif = Path(cudem_tif)
    if not pred_tif.exists():
        raise FileNotFoundError(f"Prediction TIF not found: {pred_tif}")
    if not cudem_tif.exists():
        raise FileNotFoundError(f"Aligned CUDEM TIF not found: {cudem_tif}")

    with rasterio.open(pred_tif) as pred_src, rasterio.open(cudem_tif) as cudem_src:
        if pred_src.crs != cudem_src.crs:
            raise ValueError(
                f"Prediction and aligned CUDEM CRS differ: {pred_src.crs} vs {cudem_src.crs}"
            )
        if pred_src.res != cudem_src.res:
            raise ValueError(
                f"Prediction and aligned CUDEM resolutions differ: {pred_src.res} vs {cudem_src.res}"
            )

        profile = pred_src.profile.copy()
        if pred_src.shape == cudem_src.shape and pred_src.transform == cudem_src.transform:
            pred = pred_src.read(1).astype("float64")
            cudem = cudem_src.read(1).astype("float64")
        else:
            left = max(pred_src.bounds.left, cudem_src.bounds.left)
            bottom = max(pred_src.bounds.bottom, cudem_src.bounds.bottom)
            right = min(pred_src.bounds.right, cudem_src.bounds.right)
            top = min(pred_src.bounds.top, cudem_src.bounds.top)
            if left >= right or bottom >= top:
                raise ValueError(f"Prediction and aligned CUDEM do not overlap: {pred_tif} vs {cudem_tif}")
            bounds = (left, bottom, right, top)
            pred = _read_overlap(pred_src, bounds, pred_src.transform)
            cudem = _read_overlap(cudem_src, bounds, cudem_src.transform)
            if pred.shape != cudem.shape:
                rows = min(pred.shape[0], cudem.shape[0])
                cols = min(pred.shape[1], cudem.shape[1])
                pred = pred[:rows, :cols]
                cudem = cudem[:rows, :cols]
            profile.update(
                height=pred.shape[0],
                width=pred.shape[1],
                transform=pred_src.window_transform(from_bounds(*bounds, transform=pred_src.transform).round_offsets().round_lengths()),
            )

        pred = _clean_nodata(pred, pred_src.nodata)
        cudem = _clean_nodata(cudem, cudem_src.nodata)

    return pred, -cudem, profile


def _masked_depth(arr: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_invalid(np.asarray(arr, dtype="float64"))


def _hillshade(arr: np.ndarray, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    data = np.asarray(arr, dtype="float64")
    finite = np.isfinite(data)
    if finite.sum() < 2:
        return np.ones_like(data, dtype="float64") * 0.65
    fill = float(np.nanmedian(data[finite]))
    smooth = np.where(finite, data, fill)
    dy, dx = np.gradient(smooth)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(altitude)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shade = (shade - np.nanmin(shade)) / (np.nanmax(shade) - np.nanmin(shade) + 1e-12)
    return np.where(finite, shade, np.nan)


def _axes_extent(profile: dict, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    transform = profile.get("transform")
    if transform is None:
        rows, cols = shape
        return 0.0, float(cols), float(rows), 0.0
    rows, cols = shape
    left = transform.c
    top = transform.f
    right = left + transform.a * cols
    bottom = top + transform.e * rows
    return left / 1000.0, right / 1000.0, bottom / 1000.0, top / 1000.0


def _add_scale_bar(ax, extent, length_km: float | None = None) -> None:
    xmin, xmax, ymin, ymax = extent
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        return
    if length_km is None:
        candidates = np.array([1, 2, 5, 10, 20, 50], dtype="float64")
        length_km = float(candidates[candidates <= width / 4].max(initial=1))
    x0 = xmin + width * 0.06
    y0 = ymin + height * 0.07
    ax.plot([x0, x0 + length_km], [y0, y0], color="black", linewidth=2.2, solid_capstyle="butt")
    ax.text(x0 + length_km / 2, y0 + height * 0.025, f"{length_km:g} km",
            ha="center", va="bottom", fontsize=7.5, color="black")


def _add_north_arrow(ax) -> None:
    ax.annotate(
        "N",
        xy=(0.93, 0.18),
        xytext=(0.93, 0.07),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", linewidth=1.2, color="black"),
    )


def _format_map_axis(ax, extent) -> None:
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km)", fontsize=8)
    ax.set_ylabel("Northing (km)", fontsize=8)
    ax.tick_params(labelsize=7, length=2.5, width=0.6)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def plot_publication_depth_map(
    arr: np.ndarray,
    profile: dict,
    out_png: Path,
    title: str,
    subtitle: str | None = None,
) -> None:
    """Create a publication-style bathymetry map with hillshade and scale context."""
    out_png = Path(out_png)
    extent = _axes_extent(profile, arr.shape)
    cmap = BATHYMETRY_CMAP.copy()
    cmap.set_bad("#f0f0f0")

    fig, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
    depth = _masked_depth(arr)
    im = ax.imshow(depth, cmap=cmap, vmin=DEPTH_VMIN, vmax=DEPTH_VMAX, extent=extent, origin="upper")
    shade = np.ma.masked_invalid(_hillshade(arr))
    ax.imshow(shade, cmap="gray", alpha=0.22, extent=extent, origin="upper", vmin=0, vmax=1)
    _format_map_axis(ax, extent)
    _add_scale_bar(ax, extent)
    _add_north_arrow(ax)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if subtitle:
        ax.text(0.01, 0.99, subtitle, transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.5))
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Depth (m)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_publication_dem_comparison(
    region: str,
    pred: np.ndarray,
    ref: np.ndarray,
    err: np.ndarray,
    profile: dict,
    out_png: Path,
) -> None:
    """Create a journal-style predicted/CUDEM/error comparison panel."""
    extent = _axes_extent(profile, pred.shape)
    depth_cmap = BATHYMETRY_CMAP.copy()
    depth_cmap.set_bad("#f0f0f0")
    err_cmap = plt.get_cmap("RdBu_r").copy()
    err_cmap.set_bad("#f0f0f0")

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2), constrained_layout=True)
    panels = [
        ("a", "CatBoost predicted depth", pred),
        ("b", "CUDEM reference depth", ref),
    ]
    depth_images = []
    for ax, (letter, title, arr) in zip(axes[:2], panels):
        im = ax.imshow(_masked_depth(arr), cmap=depth_cmap, vmin=DEPTH_VMIN, vmax=DEPTH_VMAX,
                       extent=extent, origin="upper")
        shade = np.ma.masked_invalid(_hillshade(arr))
        ax.imshow(shade, cmap="gray", alpha=0.20, extent=extent, origin="upper", vmin=0, vmax=1)
        depth_images.append(im)
        _format_map_axis(ax, extent)
        _add_scale_bar(ax, extent)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
        ax.text(0.02, 0.98, letter, transform=ax.transAxes, ha="left", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0))

    norm = TwoSlopeNorm(vmin=-ERROR_VMAX, vcenter=0.0, vmax=ERROR_VMAX)
    err_im = axes[2].imshow(np.ma.masked_invalid(err), cmap=err_cmap, norm=norm, extent=extent, origin="upper")
    _format_map_axis(axes[2], extent)
    _add_scale_bar(axes[2], extent)
    axes[2].set_title("Prediction error", fontsize=9, fontweight="bold", pad=5)
    axes[2].text(0.02, 0.98, "c", transform=axes[2].transAxes, ha="left", va="top",
                 fontsize=10, fontweight="bold",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0))

    cbar_depth = fig.colorbar(depth_images[0], ax=axes[:2], fraction=0.032, pad=0.02)
    cbar_depth.set_label("Depth (m)", fontsize=8)
    cbar_depth.ax.tick_params(labelsize=7)
    cbar_err = fig.colorbar(err_im, ax=axes[2], fraction=0.046, pad=0.02)
    cbar_err.set_label("Prediction - CUDEM (m)", fontsize=8)
    cbar_err.ax.tick_params(labelsize=7)
    fig.suptitle(region, fontsize=11, fontweight="bold")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def density_scatter(x, y, metrics, out_png, title, xlabel, ylabel,
                    vmin=0.0, vmax=15.0):
    """Draw a density-colored scatter plot with fixed axes."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal")

    ax.plot(
        [vmin, vmax],
        [vmin, vmax],
        color="black",
        linestyle="--",
        linewidth=1.6,
        zorder=3,
    )

    if len(x) > 0:
        try:
            from scipy.stats import gaussian_kde

            sample_idx = np.arange(len(x))
            if len(x) > 50000:
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(len(x), 50000, replace=False)
            xs, ys = x[sample_idx], y[sample_idx]
            kde = gaussian_kde(np.vstack([xs, ys]))
            z = kde(np.vstack([xs, ys]))
            order = z.argsort()
            ax.scatter(
                xs[order],
                ys[order],
                c=z[order],
                s=8,
                cmap="jet",
                zorder=2,
                edgecolors="none",
            )
        except Exception:
            ax.scatter(x, y, s=6, color="steelblue", alpha=0.4, zorder=2)

    if len(x) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.array([vmin, vmax], dtype="float64")
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color="red", linewidth=2.0, zorder=4)

    text = (
        f"N = {metrics['N']}\n"
        f"$R^2$ = {metrics['R2']:.2f}\n"
        f"MAE = {metrics['MAE']:.2f} m\n"
        f"RMSE = {metrics['RMSE']:.2f} m\n"
        f"Bias = {metrics['Bias']:+.2f} m"
    )
    ax.text(
        0.97,
        0.05,
        text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="black", alpha=0.85),
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
