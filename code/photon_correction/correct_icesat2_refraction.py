import csv
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod
from sklearn.linear_model import LinearRegression, RANSACRegressor


SURFACE_LABEL = 1
BATHY_LABEL = 3

ALONG_TRACK_COL = "Along_Track_Dist"
ELEVATION_COL = "Geoid_Corrected_Ortho_Height"
LABEL_COL = "pred_label"
LAT_COL = "Latitude"
LON_COL = "Longitude"
REF_ELEV_COL = "ref_elev"
REF_AZIMUTH_COL = "ref_azimuth"

SURFACE_ELEV_COL = "Surface_Local_Elev"
SURFACE_X_COL = "Surface_Local_Along_Track"
APPARENT_DEPTH_COL = "Depth_Apparent"
DEPTH_COL = "Depth"
METHOD_COL = "Depth_Correction_Method"

BASE_DATA_DIR = Path(r"H:\ATL24\PhisicKPConvNet\validation_area\data")
AREAS = ["Florida Bay"]
INPUT_SUBDIR = Path("Predictions_1")
BACKUP_SUBDIR = "backup_before_multi_model_refraction"
REPORT_PATH = Path(
    r"H:\ATL24\PhisicKPConvNet\Sentinel-2\ICEsat-2_Sentinel-2_Match\ransac_refraction_multi_model_report.csv"
)

N_AIR = 1.00029
N_WATER = 1.34116
REFRACTION_RATIO = N_AIR / N_WATER
LOCAL_SURFACE_NEIGHBORS = 50
METHOD_NAME = "ransac_50_linear_surface_refraction"
GEOD = Geod(ellps="WGS84")

REQUIRED_COLS = [
    LABEL_COL,
    ALONG_TRACK_COL,
    ELEVATION_COL,
    LAT_COL,
    LON_COL,
    REF_ELEV_COL,
    REF_AZIMUTH_COL,
]

OUTPUT_COLS = [
    SURFACE_ELEV_COL,
    SURFACE_X_COL,
    APPARENT_DEPTH_COL,
    DEPTH_COL,
    METHOD_COL,
    "Corrected_Latitude",
    "Corrected_Longitude",
    "Corrected_Geoid_Corrected_Ortho_Height",
]

DIAGNOSTIC_OUTPUT_COLS = [
    "Surface_Slope_Angle",
    "Incidence_Angle",
    "Refraction_Angle",
    "True_Path_Angle",
    "Offset_Direction_Angle",
    "Refraction_Path_Difference",
    "Refraction_Delta_D",
    "Refraction_Delta_Z",
    "Refraction_Delta_N",
    "Refraction_Delta_E",
]

WRITE_DIAGNOSTIC_COLS = False

MODEL_LABEL_COLS = [
    "Pred_PointNetPP",
    "Pred_PointNeXt",
    "Pred_KPConv",
    "Pred_DGCNN",
    "Pred_Proposed_method",
]


def _suffix_from_label_col(label_col):
    if label_col == LABEL_COL:
        return ""
    if label_col.startswith("Pred_"):
        return label_col[len("Pred_"):]
    return label_col


def _out_col(base_col, label_col):
    suffix = _suffix_from_label_col(label_col)
    if not suffix:
        return base_col
    return f"{base_col}_{suffix}"


def _output_cols_for_label(label_col):
    cols = list(OUTPUT_COLS)
    if WRITE_DIAGNOSTIC_COLS:
        cols.extend(DIAGNOSTIC_OUTPUT_COLS)
    return [_out_col(col, label_col) for col in cols]


def _method_col_for_label(label_col):
    return _out_col(METHOD_COL, label_col)


def _set_if_present(df, idx, col, value):
    if col in df.columns:
        df.at[idx, col] = value


def _available_label_cols(df, requested_cols=None, include_pred_label=False):
    cols = []
    if include_pred_label and LABEL_COL in df.columns:
        cols.append(LABEL_COL)
    candidates = requested_cols or MODEL_LABEL_COLS
    for col in candidates:
        if col in df.columns and col not in cols:
            cols.append(col)
    if not cols:
        cols = [col for col in df.columns if col.startswith("Pred_")]
    return cols


def _safe_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _nearest_surface_indices(surface_x, target_x, count):
    n = surface_x.size
    right = int(np.searchsorted(surface_x, target_x, side="left"))
    left = right - 1
    indices = []
    while len(indices) < count and (left >= 0 or right < n):
        if left < 0:
            indices.append(right)
            right += 1
        elif right >= n:
            indices.append(left)
            left -= 1
        elif abs(surface_x[left] - target_x) <= abs(surface_x[right] - target_x):
            indices.append(left)
            left -= 1
        else:
            indices.append(right)
            right += 1
    return np.asarray(indices, dtype=int)


def _fit_ransac_linear_surface(surface_x, surface_z, target_x):
    if surface_x.size < 2:
        return None

    local_idx = _nearest_surface_indices(
        surface_x,
        target_x,
        min(LOCAL_SURFACE_NEIGHBORS, surface_x.size),
    )
    local_x = surface_x[local_idx]
    local_z = surface_z[local_idx]
    if local_x.size < 2 or np.unique(local_x).size < 2:
        return None

    nearest = local_idx[0]
    x_s = float(surface_x[nearest])
    z_s = float(surface_z[nearest])
    x_centered = (local_x - x_s).reshape(-1, 1)

    try:
        try:
            model = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=2,
                max_trials=30,
                random_state=0,
            )
        except TypeError:
            model = RANSACRegressor(
                base_estimator=LinearRegression(),
                min_samples=2,
                max_trials=30,
                random_state=0,
            )
        model.fit(x_centered, local_z)
        estimator = model.estimator_
        slope = float(estimator.coef_[0])
    except (ValueError, AttributeError, np.linalg.LinAlgError):
        return None

    if not np.isfinite(slope):
        return None

    return {
        "x_s": x_s,
        "z_s": z_s,
        "slope": slope,
        "surface_point_count": int(local_x.size),
    }


def _surface_angle_for_piecewise(phi):
    phi_piece = float(phi)
    if phi_piece < 0:
        phi_piece += np.pi
    return phi_piece


def _chen_piecewise_angles(alpha, phi):
    phi_piece = _surface_angle_for_piecewise(phi)
    if alpha > phi_piece:
        branch = 1
        theta = alpha - phi_piece
    elif phi_piece <= alpha + np.pi / 2.0:
        branch = 2
        theta = phi_piece - alpha
    elif phi_piece <= np.pi:
        branch = 3
        theta = np.pi - phi_piece + alpha
    else:
        return None

    theta = float(np.clip(theta, 0.0, np.pi / 2.0))
    beta = float(np.arcsin(np.clip(REFRACTION_RATIO * np.sin(theta), -1.0, 1.0)))

    if branch == 1:
        tau = phi_piece + beta
        gamma = np.pi / 2.0 + alpha + tau
    elif branch == 2:
        tau = phi_piece - beta
        gamma = np.pi / 2.0 + alpha - tau
    else:
        tau = phi_piece + beta - np.pi
        gamma = np.pi / 2.0 + alpha + tau

    return {
        "theta": theta,
        "beta": beta,
        "tau": float(tau),
        "gamma": float(gamma),
        "phi_piece": phi_piece,
    }


def _offset_lat_lon(lat, lon, delta_n, delta_e):
    distance = float(np.hypot(delta_n, delta_e))
    if distance == 0.0:
        return float(lat), float(lon)
    azimuth_deg = float(np.degrees(np.arctan2(delta_e, delta_n)))
    lon_c, lat_c, _ = GEOD.fwd(float(lon), float(lat), azimuth_deg, distance)
    return float(lat_c), float(lon_c)


def _correct_point(row, surface_x, surface_z, surface_cache):
    x_b = float(row[ALONG_TRACK_COL])
    z_b = float(row[ELEVATION_COL])
    lat = float(row[LAT_COL])
    lon = float(row[LON_COL])
    ref_elev = float(row[REF_ELEV_COL])
    kappa = float(row[REF_AZIMUTH_COL])

    if not all(np.isfinite(v) for v in [x_b, z_b, lat, lon, ref_elev, kappa]):
        return {"method": f"{METHOD_NAME}; invalid_missing_angle"}

    cache_key = round(x_b, 6)
    if cache_key not in surface_cache:
        surface_cache[cache_key] = _fit_ransac_linear_surface(surface_x, surface_z, x_b)
    surface = surface_cache[cache_key]
    if surface is None:
        return {"method": f"{METHOD_NAME}; invalid_ransac_surface"}

    x_s = surface["x_s"]
    z_s = surface["z_s"]
    apparent_depth = z_s - z_b
    if apparent_depth <= 0 or not np.isfinite(apparent_depth):
        return {
            "method": f"{METHOD_NAME}; invalid_nonpositive_depth",
            "surface_x": x_s,
            "surface_z": z_s,
            "apparent_depth": apparent_depth,
        }

    alpha = np.pi / 2.0 - ref_elev
    if not np.isfinite(alpha):
        return {
            "method": f"{METHOD_NAME}; invalid_missing_angle",
            "surface_x": x_s,
            "surface_z": z_s,
            "apparent_depth": apparent_depth,
        }
    alpha = float(np.clip(alpha, 0.0, np.pi / 2.0))

    phi = float(np.arctan(surface["slope"]))
    angles = _chen_piecewise_angles(alpha, phi)
    if angles is None:
        return {
            "method": f"{METHOD_NAME}; invalid_angle_geometry",
            "surface_x": x_s,
            "surface_z": z_s,
            "apparent_depth": apparent_depth,
            "phi": phi,
        }

    s_path = float(np.hypot(x_b - x_s, z_b - z_s))
    if s_path <= 0 or not np.isfinite(s_path):
        return {
            "method": f"{METHOD_NAME}; invalid_zero_path",
            "surface_x": x_s,
            "surface_z": z_s,
            "apparent_depth": apparent_depth,
            "phi": phi,
        }

    r_path = REFRACTION_RATIO * s_path
    tau = angles["tau"]
    p_diff = float(
        np.linalg.norm(
            np.array([s_path, 0.0])
            - r_path * np.array([np.cos(alpha - tau), np.sin(alpha - tau)])
        )
    )
    gamma = angles["gamma"]
    delta_d = float(p_diff * np.cos(gamma))
    delta_z = float(p_diff * np.sin(gamma))
    delta_n = float(delta_d * np.cos(kappa))
    delta_e = float(delta_d * np.sin(kappa))

    z_c = z_b + delta_z
    corrected_depth = z_s - z_c
    if corrected_depth <= 0 or not np.isfinite(corrected_depth):
        method = f"{METHOD_NAME}; invalid_corrected_depth"
    else:
        method = METHOD_NAME

    lat_c, lon_c = _offset_lat_lon(lat, lon, delta_n, delta_e)

    return {
        "method": method,
        "surface_x": x_s,
        "surface_z": z_s,
        "apparent_depth": apparent_depth,
        "corrected_depth": corrected_depth,
        "corrected_z": z_c,
        "corrected_lat": lat_c,
        "corrected_lon": lon_c,
        "phi": phi,
        "theta": angles["theta"],
        "beta": angles["beta"],
        "tau": tau,
        "gamma": gamma,
        "p_diff": p_diff,
        "delta_d": delta_d,
        "delta_z": delta_z,
        "delta_n": delta_n,
        "delta_e": delta_e,
    }


def _process_label_source(df, label_col):
    method_col = _method_col_for_label(label_col)
    surface_mask = df[label_col] == SURFACE_LABEL
    seabed_mask = df[label_col] == BATHY_LABEL
    surface_df = df.loc[surface_mask, [ALONG_TRACK_COL, ELEVATION_COL]].dropna()
    if surface_df.empty:
        return {
            "status": "failed",
            "message": f"no surface photons with {label_col} == {SURFACE_LABEL}",
            "seabed_rows": int(seabed_mask.sum()),
            "valid_depth_rows": 0,
            "invalid_depth_rows": int(seabed_mask.sum()),
        }

    surface_values = surface_df[[ALONG_TRACK_COL, ELEVATION_COL]].to_numpy(dtype=float)
    valid_surface = np.isfinite(surface_values[:, 0]) & np.isfinite(surface_values[:, 1])
    surface_values = surface_values[valid_surface]
    if surface_values.size == 0:
        return {
            "status": "failed",
            "message": f"no finite surface photons with {label_col} == {SURFACE_LABEL}",
            "seabed_rows": int(seabed_mask.sum()),
            "valid_depth_rows": 0,
            "invalid_depth_rows": int(seabed_mask.sum()),
        }

    order = np.argsort(surface_values[:, 0])
    surface_values = surface_values[order]
    surface_x = surface_values[:, 0]
    surface_z = surface_values[:, 1]

    col_surface_elev = _out_col(SURFACE_ELEV_COL, label_col)
    col_surface_x = _out_col(SURFACE_X_COL, label_col)
    col_apparent_depth = _out_col(APPARENT_DEPTH_COL, label_col)
    col_depth = _out_col(DEPTH_COL, label_col)
    col_corrected_lat = _out_col("Corrected_Latitude", label_col)
    col_corrected_lon = _out_col("Corrected_Longitude", label_col)
    col_corrected_z = _out_col("Corrected_Geoid_Corrected_Ortho_Height", label_col)
    col_phi = _out_col("Surface_Slope_Angle", label_col)
    col_theta = _out_col("Incidence_Angle", label_col)
    col_beta = _out_col("Refraction_Angle", label_col)
    col_tau = _out_col("True_Path_Angle", label_col)
    col_gamma = _out_col("Offset_Direction_Angle", label_col)
    col_p_diff = _out_col("Refraction_Path_Difference", label_col)
    col_delta_d = _out_col("Refraction_Delta_D", label_col)
    col_delta_z = _out_col("Refraction_Delta_Z", label_col)
    col_delta_n = _out_col("Refraction_Delta_N", label_col)
    col_delta_e = _out_col("Refraction_Delta_E", label_col)

    valid_depth_rows = 0
    invalid_depth_rows = 0
    surface_cache = {}

    for idx, row in df.loc[seabed_mask].iterrows():
        result = _correct_point(row, surface_x, surface_z, surface_cache)
        df.at[idx, method_col] = result["method"]
        df.at[idx, col_surface_x] = result.get("surface_x", np.nan)
        df.at[idx, col_surface_elev] = result.get("surface_z", np.nan)
        df.at[idx, col_apparent_depth] = result.get("apparent_depth", np.nan)
        _set_if_present(df, idx, col_phi, result.get("phi", np.nan))
        _set_if_present(df, idx, col_theta, result.get("theta", np.nan))
        _set_if_present(df, idx, col_beta, result.get("beta", np.nan))
        _set_if_present(df, idx, col_tau, result.get("tau", np.nan))
        _set_if_present(df, idx, col_gamma, result.get("gamma", np.nan))
        _set_if_present(df, idx, col_p_diff, result.get("p_diff", np.nan))
        _set_if_present(df, idx, col_delta_d, result.get("delta_d", np.nan))
        _set_if_present(df, idx, col_delta_z, result.get("delta_z", np.nan))
        _set_if_present(df, idx, col_delta_n, result.get("delta_n", np.nan))
        _set_if_present(df, idx, col_delta_e, result.get("delta_e", np.nan))

        corrected_depth = result.get("corrected_depth", np.nan)
        if np.isfinite(corrected_depth) and corrected_depth > 0 and "invalid" not in result["method"]:
            df.at[idx, col_depth] = corrected_depth
            df.at[idx, col_corrected_z] = result["corrected_z"]
            df.at[idx, col_corrected_lat] = result["corrected_lat"]
            df.at[idx, col_corrected_lon] = result["corrected_lon"]
            valid_depth_rows += 1
        else:
            invalid_depth_rows += 1

    return {
        "status": "success",
        "message": "ok",
        "seabed_rows": int(seabed_mask.sum()),
        "valid_depth_rows": valid_depth_rows,
        "invalid_depth_rows": invalid_depth_rows,
    }


def process_csv(input_csv, backup_dir, label_cols=None, include_pred_label=False):
    df = pd.read_csv(input_csv, low_memory=False)
    available_label_cols = _available_label_cols(
        df,
        requested_cols=label_cols,
        include_pred_label=include_pred_label,
    )
    required_cols = [
        ALONG_TRACK_COL,
        ELEVATION_COL,
        LAT_COL,
        LON_COL,
        REF_ELEV_COL,
        REF_AZIMUTH_COL,
    ] + available_label_cols
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return [{
            "label_col": "",
            "status": "failed",
            "message": f"missing columns: {missing}",
            "input_rows": len(df),
            "seabed_rows": 0,
            "valid_depth_rows": 0,
            "invalid_depth_rows": 0,
        }]
    if not available_label_cols:
        return [{
            "label_col": "",
            "status": "failed",
            "message": "no prediction label columns found",
            "input_rows": len(df),
            "seabed_rows": 0,
            "valid_depth_rows": 0,
            "invalid_depth_rows": 0,
        }]

    df = _safe_numeric(df, required_cols)

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / input_csv.name
    if not backup_path.exists():
        shutil.copy2(input_csv, backup_path)

    init_cols = {}
    drop_cols = set()
    for label_col in available_label_cols:
        method_col = _method_col_for_label(label_col)
        for col in _output_cols_for_label(label_col):
            init_cols[col] = "" if col == method_col else np.nan
            drop_cols.add(col)
        if not WRITE_DIAGNOSTIC_COLS:
            for col in DIAGNOSTIC_OUTPUT_COLS:
                drop_cols.add(_out_col(col, label_col))
    if init_cols:
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
        df = pd.concat([df, pd.DataFrame(init_cols, index=df.index)], axis=1)

    results = []
    for label_col in available_label_cols:
        result = _process_label_source(df, label_col)
        result.update({
            "label_col": label_col,
            "input_rows": len(df),
        })
        results.append(result)

    df.to_csv(input_csv, index=False, encoding="utf-8-sig")
    return results


def iter_inputs(areas, input_subdir, limit=None, smallest_first=False):
    count = 0
    for area in areas:
        if limit is not None and count >= limit:
            break
        input_dir = BASE_DATA_DIR / area / input_subdir
        if not input_dir.exists():
            print(f"[SKIP] Missing enhanced folder: {input_dir}")
            continue
        input_files = list(input_dir.glob("*.csv"))
        if smallest_first:
            input_files = sorted(input_files, key=lambda p: p.stat().st_size)
        else:
            input_files = sorted(input_files)
        for input_csv in input_files:
            yield area, input_csv, input_dir / BACKUP_SUBDIR
            count += 1
            if limit is not None and count >= limit:
                break


def write_report(rows):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "area",
        "csv_name",
        "input_csv",
        "label_col",
        "status",
        "message",
        "input_rows",
        "seabed_rows",
        "valid_depth_rows",
        "invalid_depth_rows",
    ]
    with open(REPORT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply identical RANSAC refraction depth correction to multiple "
            "model prediction columns in-place."
        )
    )
    parser.add_argument("--areas", nargs="+", default=AREAS)
    parser.add_argument("--input-subdir", default=str(INPUT_SUBDIR))
    parser.add_argument(
        "--label-cols",
        nargs="+",
        default=MODEL_LABEL_COLS,
        help="Prediction columns to correct. Missing columns are ignored.",
    )
    parser.add_argument(
        "--include-pred-label",
        action="store_true",
        help="Also recompute the original pred_label correction columns.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of CSV files to process for a quick test.",
    )
    parser.add_argument(
        "--smallest-first",
        action="store_true",
        help="Process smaller CSV files first; useful with --limit for smoke tests.",
    )
    parser.add_argument(
        "--surface-neighbors",
        type=int,
        default=LOCAL_SURFACE_NEIGHBORS,
        help="Number of nearest sea-surface photons used for local RANSAC fitting.",
    )
    parser.add_argument(
        "--write-diagnostic-cols",
        action="store_true",
        help="Also write refraction angle/path diagnostic columns for each model.",
    )
    return parser.parse_args()


def main():
    global LOCAL_SURFACE_NEIGHBORS, WRITE_DIAGNOSTIC_COLS
    args = parse_args()
    LOCAL_SURFACE_NEIGHBORS = max(2, int(args.surface_neighbors))
    WRITE_DIAGNOSTIC_COLS = bool(args.write_diagnostic_cols)
    reports = []
    items = list(iter_inputs(
        args.areas,
        Path(args.input_subdir),
        args.limit,
        smallest_first=args.smallest_first,
    ))
    for idx, (area, input_csv, backup_dir) in enumerate(items, 1):
        print(f"\n[{idx}/{len(items)}] {area}: {input_csv.name}", flush=True)
        results = process_csv(
            input_csv,
            backup_dir,
            label_cols=args.label_cols,
            include_pred_label=args.include_pred_label,
        )
        for result in results:
            result.update({
                "area": area,
                "csv_name": input_csv.name,
                "input_csv": str(input_csv),
            })
            reports.append(result)
            label = result.get("label_col", "")
            print(
                f"[{result['status']}] {label}: seabed={result['seabed_rows']}, "
                f"valid={result['valid_depth_rows']}, invalid={result['invalid_depth_rows']}: "
                f"{result['message']}",
                flush=True,
            )

    write_report(reports)
    ok = sum(1 for row in reports if row["status"] == "success")
    print(f"\nDone. Successful files: {ok}/{len(reports)}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
