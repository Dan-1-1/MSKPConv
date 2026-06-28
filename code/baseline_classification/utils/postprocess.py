"""Gaussian-KDE-based sea-surface estimation and label refinement.

Post-processing rules (operating in physical coordinates, metres):
1. Take all photons currently predicted as surface (label == 1).
2. Run one global 1-D Gaussian KDE over their geoid-corrected height z and
   take the mode as the file-level sea-surface height z_surf.
3. For each photon classified as signal (pred != noise/0):
        - if abs(z_i - z_surf) <= surface_band_m  -> relabel as surface (1)
        - if z_i > z_surf + land_offset_m         -> relabel as land    (2)
        - if z_i <= z_surf - seabed_offset_m      -> relabel as seabed   (3)
        - otherwise: keep current prediction.
"""
import numpy as np


def _kde_peak_1d(samples: np.ndarray, bw: float, grid_size: int = 200) -> float:
    """Find the mode of a 1-D Gaussian KDE on ``samples`` with bandwidth ``bw``."""
    samples = np.asarray(samples, dtype=np.float64)
    n = samples.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(samples[0])
    z_min = float(samples.min()) - 3.0 * bw
    z_max = float(samples.max()) + 3.0 * bw
    if z_max - z_min < 1e-9:
        return float(samples.mean())
    grid = np.linspace(z_min, z_max, grid_size)
    diff = (grid[:, None] - samples[None, :]) / bw
    kde = np.exp(-0.5 * diff * diff).sum(axis=1)
    return float(grid[int(np.argmax(kde))])


# def estimate_sea_surface(
#     x: np.ndarray,
#     z: np.ndarray,
#     surface_mask: np.ndarray,
#     window_m: float = 50.0,
#     bw: float = 0.3,
#     min_points_per_window: int = 5,
#     smooth_kernel: int = 5,
# ) -> np.ndarray:
#     """Estimate one file-level sea-surface height for every photon."""
#     _ = x, window_m, min_points_per_window, smooth_kernel
#     z = np.asarray(z, dtype=np.float64)
#     n = z.size
#     if n == 0:
#         return np.zeros(0, dtype=np.float64)

#     if not surface_mask.any():
#         return np.full(n, float(np.median(z)), dtype=np.float64)

#     sz = z[surface_mask]
#     peak = _kde_peak_1d(sz, bw)
#     return np.full(n, peak, dtype=np.float64)
def estimate_sea_surface(
    x: np.ndarray,
    z: np.ndarray,
    surface_mask: np.ndarray,
    window_m: float = 1000.0,
    bw: float = 0.3,
    min_points_per_window: int = 5,
    smooth_kernel: int = 5,
) -> np.ndarray:
    """Estimate sea-surface height by fixed along-track windows.

    The along-track axis is split into non-overlapping windows with width
    ``window_m`` metres. All photons in the same window share one sea-surface
    height, estimated by 1-D KDE from photons currently predicted as surface
    inside that window.

    If a window has too few surface photons, it falls back to the global
    file-level sea-surface estimate.
    """
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    surface_mask = np.asarray(surface_mask, dtype=bool)

    n = z.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(z)
    if not valid.any():
        return np.zeros(n, dtype=np.float64)

    valid_surface = valid & surface_mask
    if valid_surface.any():
        global_peak = _kde_peak_1d(z[valid_surface], bw)
    else:
        global_peak = float(np.nanmedian(z[valid]))

    z_surf = np.full(n, global_peak, dtype=np.float64)

    x_min = float(np.nanmin(x[valid]))
    x_max = float(np.nanmax(x[valid]))
    if window_m <= 0 or x_max <= x_min:
        return z_surf

    # Non-overlapping 1000 m along-track windows:
    # [x_min, x_min+1000), [x_min+1000, x_min+2000), ...
    window_id = np.floor((x - x_min) / window_m).astype(np.int64)

    for wid in np.unique(window_id[valid]):
        in_window = valid & (window_id == wid)
        surface_in_window = in_window & surface_mask

        if np.count_nonzero(surface_in_window) >= min_points_per_window:
            local_peak = _kde_peak_1d(z[surface_in_window], bw)
        else:
            local_peak = global_peak

        z_surf[in_window] = local_peak

    # Optional smoothing across adjacent photons after assigning window-level
    # surfaces. If you want each 1000 m block to remain strictly constant,
    # set smooth_kernel=1 or remove this block.
    if smooth_kernel is not None and smooth_kernel > 1:
        order = np.argsort(x[valid])
        valid_idx = np.where(valid)[0][order]

        k = int(smooth_kernel)
        if k % 2 == 0:
            k += 1

        pad = k // 2
        values = z_surf[valid_idx]
        padded = np.pad(values, pad_width=pad, mode="edge")
        kernel = np.ones(k, dtype=np.float64) / k
        smoothed = np.convolve(padded, kernel, mode="valid")
        z_surf[valid_idx] = smoothed

    return z_surf
def apply_surface_postprocess(
    x: np.ndarray,
    z: np.ndarray,
    pred: np.ndarray,
    surface_band_m: float = 0.5,
    seabed_offset_m: float = 1.0,
    window_m: float = 1000.0,
    bw: float = 0.3,
    surface_class: int = 1,
    land_class: int = 2,
    seabed_class: int = 3,
    noise_class: int = 0,
    land_offset_m: float = 1.0,
) -> np.ndarray:
    """Refine ``pred`` using the file-level KDE-estimated sea surface."""
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.int64)
    if pred.size == 0:
        return pred.copy()

    pred_out = pred.copy()
    surface_mask = pred == surface_class
    z_surf = estimate_sea_surface(x, z, surface_mask, window_m=window_m, bw=bw)

    signal_mask = pred != noise_class
    dz = z - z_surf

    in_band = signal_mask & (np.abs(dz) <= surface_band_m)
    pred_out[in_band] = surface_class

    land = signal_mask & (dz > land_offset_m)
    pred_out[land] = land_class

    deep = signal_mask & (dz <= -seabed_offset_m)
    pred_out[deep] = seabed_class

    return pred_out
