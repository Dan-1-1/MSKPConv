"""Fast multi-label ICESat-2 bathymetry refraction correction.

This implementation keeps the same refraction geometry as the reviewer
release script, but replaces a per-photon sklearn RANSAC fit with a robust
linear surface evaluated on a fixed 2 m along-track grid.  Surface height and
slope are interpolated to all seabed photons and the correction is vectorized.
Files are independent and can therefore be processed with multiple processes.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod


BASE_DATA_DIR = Path(r"H:\ATL24\PhisicKPConvNet\validation_area\data")
AREAS = ["Florida Bay", "Key Largo", "Key West", "Marathon"]
INPUT_SUBDIR = Path("Predictions_1")
BACKUP_SUBDIR = "backup_before_multi_model_refraction"
REPORT_PATH = Path(
    r"H:\ATL24\PhisicKPConvNet\Sentinel-2\ICEsat-2_Sentinel-2_Match"
) / "ransac_refraction_multi_model_report_2m_grid.csv"

SURFACE_LABEL = 1
BATHY_LABEL = 3
GRID_SPACING_M = 2.0
LOCAL_SURFACE_NEIGHBORS = 50
N_AIR = 1.00029
N_WATER = 1.34116
REFRACTION_RATIO = N_AIR / N_WATER
METHOD_NAME = "grid2m_robust_linear_surface_refraction"
GEOD = Geod(ellps="WGS84")

X = "Along_Track_Dist"
Z = "Geoid_Corrected_Ortho_Height"
LAT = "Latitude"
LON = "Longitude"
REF_ELEV = "ref_elev"
REF_AZ = "ref_azimuth"

LABELS = [
    "Pred_PointNetPP",
    "Pred_PointNeXt",
    "Pred_KPConv",
    "Pred_DGCNN",
    "Pred_Proposed_method",
    "Song2024_Project_Class",
    "ATL24_Label",
]

OUTPUT_BASE = [
    "Surface_Local_Elev",
    "Surface_Local_Along_Track",
    "Depth_Apparent",
    "Depth",
    "Depth_Correction_Method",
    "Corrected_Latitude",
    "Corrected_Longitude",
    "Corrected_Geoid_Corrected_Ortho_Height",
]


def suffix(label: str) -> str:
    return label[5:] if label.startswith("Pred_") else label


def out_col(base: str, label: str) -> str:
    return f"{base}_{suffix(label)}"


def output_columns(label: str) -> list[str]:
    return [out_col(c, label) for c in OUTPUT_BASE]


def build_surface_grid(surface_x: np.ndarray, surface_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit robust local lines on a 2 m grid and return height and slope."""
    valid = np.isfinite(surface_x) & np.isfinite(surface_z)
    sx = np.asarray(surface_x[valid], dtype=np.float64)
    sz = np.asarray(surface_z[valid], dtype=np.float64)
    if sx.size < 2:
        raise ValueError("fewer than two finite surface photons")
    order = np.argsort(sx, kind="mergesort")
    sx, sz = sx[order], sz[order]
    keep = np.r_[True, np.diff(sx) > 0]
    sx, sz = sx[keep], sz[keep]
    if sx.size < 2:
        raise ValueError("surface photons have no distinct along-track coordinates")

    lo = math.floor(float(sx[0]) / GRID_SPACING_M) * GRID_SPACING_M
    hi = math.ceil(float(sx[-1]) / GRID_SPACING_M) * GRID_SPACING_M
    grid = np.arange(lo, hi + GRID_SPACING_M * 0.5, GRID_SPACING_M, dtype=np.float64)
    k = min(LOCAL_SURFACE_NEIGHBORS, sx.size)
    insertion = np.searchsorted(sx, grid, side="left")
    start = np.clip(insertion - k // 2, 0, sx.size - k)
    local_idx = start[:, None] + np.arange(k, dtype=np.int64)[None, :]
    lx = sx[local_idx]
    lz = sz[local_idx]

    # First least-squares line, then a MAD-based inlier refit.  This retains
    # robust surface rejection without constructing thousands of sklearn models.
    xc = lx - grid[:, None]
    xmean = xc.mean(axis=1)
    zmean = lz.mean(axis=1)
    denom = np.sum((xc - xmean[:, None]) ** 2, axis=1)
    slope = np.divide(
        np.sum((xc - xmean[:, None]) * (lz - zmean[:, None]), axis=1),
        denom,
        out=np.zeros_like(denom),
        where=denom > 1e-12,
    )
    intercept = zmean - slope * xmean
    residual = lz - (intercept[:, None] + slope[:, None] * xc)
    mad = np.median(np.abs(residual), axis=1)
    threshold = np.maximum(3.0 * mad, 0.10)
    inlier = np.abs(residual) <= threshold[:, None]
    count = inlier.sum(axis=1)
    safe_count = np.maximum(count, 2)
    xsum = np.where(inlier, xc, 0.0).sum(axis=1)
    zsum = np.where(inlier, lz, 0.0).sum(axis=1)
    xbar = xsum / safe_count
    zbar = zsum / safe_count
    dx = np.where(inlier, xc - xbar[:, None], 0.0)
    dz = np.where(inlier, lz - zbar[:, None], 0.0)
    d2 = np.sum(dx * dx, axis=1)
    slope = np.divide(np.sum(dx * dz, axis=1), d2, out=slope, where=d2 > 1e-12)
    intercept = zbar - slope * xbar
    grid_z = intercept + slope * 0.0
    bad = count < 2
    grid_z[bad] = np.interp(grid[bad], sx, sz)
    slope[bad] = 0.0
    return grid, grid_z, slope


def vector_correct(df: pd.DataFrame, label: str, grid: np.ndarray, grid_z: np.ndarray, grid_slope: np.ndarray) -> dict:
    mask_surface = df[label].to_numpy() == SURFACE_LABEL
    mask_bathy = df[label].to_numpy() == BATHY_LABEL
    seabed_idx = np.flatnonzero(mask_bathy)
    result = {
        "label_col": label,
        "status": "success",
        "message": "ok",
        "seabed_rows": int(mask_bathy.sum()),
        "valid_depth_rows": 0,
        "invalid_depth_rows": int(mask_bathy.sum()),
    }
    if not mask_surface.any():
        result.update(status="failed", message=f"no surface photons with {label} == {SURFACE_LABEL}")
        return result
    if not len(seabed_idx):
        result.update(invalid_depth_rows=0)
        return result

    x = pd.to_numeric(df[X], errors="coerce").to_numpy(np.float64)[seabed_idx]
    z = pd.to_numeric(df[Z], errors="coerce").to_numpy(np.float64)[seabed_idx]
    lat = pd.to_numeric(df[LAT], errors="coerce").to_numpy(np.float64)[seabed_idx]
    lon = pd.to_numeric(df[LON], errors="coerce").to_numpy(np.float64)[seabed_idx]
    ref_elev = pd.to_numeric(df[REF_ELEV], errors="coerce").to_numpy(np.float64)[seabed_idx]
    ref_az = pd.to_numeric(df[REF_AZ], errors="coerce").to_numpy(np.float64)[seabed_idx]
    n = len(seabed_idx)
    names = {base: out_col(base, label) for base in OUTPUT_BASE}
    for col in names.values():
        if col not in df.columns:
            df[col] = np.nan if col != names["Depth_Correction_Method"] else ""

    finite = np.isfinite(x) & np.isfinite(z) & np.isfinite(lat) & np.isfinite(lon) & np.isfinite(ref_elev) & np.isfinite(ref_az)
    method = np.full(n, f"{METHOD_NAME}; invalid_missing_angle", dtype=object)
    surface_x = np.full(n, np.nan); surface_z = np.full(n, np.nan); apparent = np.full(n, np.nan)
    corrected_depth = np.full(n, np.nan); corrected_z = np.full(n, np.nan)
    corrected_lat = np.full(n, np.nan); corrected_lon = np.full(n, np.nan)

    if finite.any():
        xf, zf = x[finite], z[finite]
        sf_x = np.interp(xf, grid, grid)
        sf_z = np.interp(xf, grid, grid_z)
        sf_slope = np.interp(xf, grid, grid_slope)
        app = sf_z - zf
        alpha = np.clip(np.pi / 2.0 - ref_elev[finite], 0.0, np.pi / 2.0)
        phi = np.arctan(sf_slope)
        phi_piece = np.where(phi < 0.0, phi + np.pi, phi)
        b1 = alpha > phi_piece
        b2 = (~b1) & (phi_piece <= alpha + np.pi / 2.0)
        b3 = (~b1) & (~b2) & (phi_piece <= np.pi)
        theta = np.where(b1, alpha - phi_piece, np.where(b2, phi_piece - alpha, np.pi - phi_piece + alpha))
        beta = np.arcsin(np.clip(REFRACTION_RATIO * np.sin(theta), -1.0, 1.0))
        tau = np.where(b1, phi_piece + beta, np.where(b2, phi_piece - beta, phi_piece + beta - np.pi))
        gamma = np.where(b1, np.pi / 2.0 + alpha + tau, np.where(b2, np.pi / 2.0 + alpha - tau, np.pi / 2.0 + alpha + tau))
        s_path = np.hypot(xf - sf_x, zf - sf_z)
        r_path = REFRACTION_RATIO * s_path
        p_diff = np.hypot(s_path - r_path * np.cos(alpha - tau), r_path * np.sin(alpha - tau))
        delta_d = p_diff * np.cos(gamma)
        delta_z = p_diff * np.sin(gamma)
        delta_n = delta_d * np.cos(ref_az[finite])
        delta_e = delta_d * np.sin(ref_az[finite])
        cz = zf + delta_z
        cd = sf_z - cz
        good = (app > 0.0) & np.isfinite(app) & (s_path > 0.0) & np.isfinite(cd) & (cd > 0.0) & (b1 | b2 | b3)
        method_f = np.full(len(xf), f"{METHOD_NAME}; invalid_corrected_depth", dtype=object)
        method_f[(app <= 0.0) | ~np.isfinite(app)] = f"{METHOD_NAME}; invalid_nonpositive_depth"
        method_f[good] = METHOD_NAME
        surface_x[finite], surface_z[finite], apparent[finite] = sf_x, sf_z, app
        corrected_depth[finite], corrected_z[finite] = cd, cz
        good_all = np.flatnonzero(finite)[good]
        method[finite] = method_f
        if good.any():
            lon_c, lat_c, _ = GEOD.fwd(lon[finite][good], lat[finite][good], np.degrees(np.arctan2(delta_e[good], delta_n[good])), np.hypot(delta_n[good], delta_e[good]))
            corrected_lat[good_all], corrected_lon[good_all] = lat_c, lon_c
        result["valid_depth_rows"] = int(good.sum())
        result["invalid_depth_rows"] = int((~good).sum())

    df.loc[seabed_idx, names["Surface_Local_Elev"]] = surface_z
    df.loc[seabed_idx, names["Surface_Local_Along_Track"]] = surface_x
    df.loc[seabed_idx, names["Depth_Apparent"]] = apparent
    df.loc[seabed_idx, names["Depth"]] = np.where(np.isfinite(corrected_depth) & (corrected_depth > 0.0), corrected_depth, np.nan)
    df.loc[seabed_idx, names["Depth_Correction_Method"]] = method
    df.loc[seabed_idx, names["Corrected_Latitude"]] = corrected_lat
    df.loc[seabed_idx, names["Corrected_Longitude"]] = corrected_lon
    df.loc[seabed_idx, names["Corrected_Geoid_Corrected_Ortho_Height"]] = np.where(np.isfinite(corrected_depth) & (corrected_depth > 0.0), corrected_z, np.nan)
    return result


def process_one(path: Path) -> list[dict]:
    df = pd.read_csv(path, low_memory=False)
    labels = [c for c in LABELS if c in df.columns]
    # Remove outputs from an earlier interrupted/slow run before recomputing.
    drop = [c for label in LABELS for c in output_columns(label) if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    for col in (X, Z, LAT, LON, REF_ELEV, REF_AZ):
        if col not in df.columns:
            return [{"csv_name": path.name, "label_col": "", "status": "failed", "message": f"missing columns: {[col]}", "input_rows": len(df)}]
    numeric = {c: pd.to_numeric(df[c], errors="coerce") for c in (X, Z, LAT, LON, REF_ELEV, REF_AZ)}
    for c, values in numeric.items():
        df[c] = values
    backup = path.parent / BACKUP_SUBDIR
    backup.mkdir(parents=True, exist_ok=True)
    bp = backup / path.name
    if not bp.exists():
        shutil.copy2(path, bp)
    rows = []
    for label in labels:
        # ATL24_Label == -1 means the track has no ATL24 reference product.
        values = pd.to_numeric(df[label], errors="coerce")
        if label == "ATL24_Label" and not values.isin([SURFACE_LABEL, BATHY_LABEL]).any():
            rows.append({"csv_name": path.name, "label_col": label, "status": "skipped", "message": "no ATL24 reference labels", "input_rows": len(df), "seabed_rows": 0, "valid_depth_rows": 0, "invalid_depth_rows": 0})
            continue
        surface = df.loc[values == SURFACE_LABEL, [X, Z]].to_numpy(float)
        try:
            grid, grid_z, grid_slope = build_surface_grid(surface[:, 0], surface[:, 1])
            result = vector_correct(df, label, grid, grid_z, grid_slope)
            result["input_rows"] = len(df)
            result["grid_spacing_m"] = GRID_SPACING_M
            rows.append(result)
        except Exception as exc:
            rows.append({"csv_name": path.name, "label_col": label, "status": "failed", "message": repr(exc), "input_rows": len(df), "seabed_rows": int((values == BATHY_LABEL).sum()), "valid_depth_rows": 0, "invalid_depth_rows": int((values == BATHY_LABEL).sum())})
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--areas", nargs="+", default=AREAS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    files = []
    for area in args.areas:
        files.extend(sorted((BASE_DATA_DIR / area / INPUT_SUBDIR).glob("*.csv")))
    if args.limit is not None:
        files = files[: args.limit]
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(process_one, p): p for p in files}
        for i, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                result = future.result(); rows.extend(result)
                print(f"[{i}/{len(files)}] {path.parent.parent.parent.name}: {path.name} labels={len(result)}", flush=True)
            except Exception as exc:
                rows.append({"csv_name": path.name, "status": "failed", "message": repr(exc)})
                print(f"[{i}/{len(files)}] FAILED {path.name}: {exc!r}", flush=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    print(f"Report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
