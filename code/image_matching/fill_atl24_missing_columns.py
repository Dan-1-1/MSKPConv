import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


BASE_DIR = Path(r"H:\ATL24\PhisicKPConvNet\validation_area\data\Florida Bay\Predictions_1")
ATL24_CSV_DIR = BASE_DIR / "ATL24_CSV"
ATL24_H5_DIR = BASE_DIR / "downloaded_atl24_h5"
REPORT_PATH = ATL24_CSV_DIR / "atl24_fill_missing_columns_report.csv"
BACKUP_DIR = ATL24_CSV_DIR / "backup_before_fill_missing_columns"

H5_TO_CSV_COLUMNS = {
    "delta_time": "delta_time",
    "lat_ph": "lat_ph",
    "lon_ph": "lon_ph",
    "ellipse_h": "h_ph",
    "x_atc": "dist_ph_along",
    "y_atc": "dist_ph_across",
    "ortho_h": "ATL24_ortho_h",
    "surface_h": "ATL24_surface_h",
    "confidence": "ATL24_confidence",
    "sigma_thu": "ATL24_sigma_thu",
    "sigma_tvu": "ATL24_sigma_tvu",
    "index_seg": "ATL24_index_seg",
    "class_ph": "ATL24_class_ph",
    "invalid_kd": "ATL24_invalid_kd",
    "invalid_wind_speed": "ATL24_invalid_wind_speed",
    "low_confidence_flag": "ATL24_low_confidence_flag",
    "night_flag": "ATL24_night_flag",
    "sensor_depth_exceeded": "ATL24_sensor_depth_exceeded",
}

ALIASES = {
    "Along_Track_Dist": "x_atc",
    "Latitude": "lat_ph",
    "Longitude": "lon_ph",
    "WGS84_Ellipsoid_Height": "ellipse_h",
    "Geoid_Corrected_Ortho_Height": "ortho_h",
    "Label": "class_ph",
}

NOT_IN_ATL24_H5 = ["ref_elev", "ref_azimuth"]


def _read_beam_table(h5_path: Path, beam: str) -> pd.DataFrame:
    with h5py.File(h5_path, "r") as h5:
        if beam not in h5:
            raise KeyError(f"beam {beam!r} not found in {h5_path.name}")
        group = h5[beam]
        if "index_ph" not in group:
            raise KeyError(f"{beam}/index_ph not found in {h5_path.name}")
        data = {"index_ph": group["index_ph"][:].astype(np.int64)}
        for h5_name in sorted(set(H5_TO_CSV_COLUMNS) | set(ALIASES.values())):
            if h5_name in group:
                data[h5_name] = group[h5_name][:]
        return pd.DataFrame(data)


def _find_h5_for_csv(df: pd.DataFrame, csv_path: Path) -> Path | None:
    if "Source_ATL24" in df.columns and df["Source_ATL24"].notna().any():
        source = str(df["Source_ATL24"].dropna().iloc[0])
        candidate = ATL24_H5_DIR / Path(source).name
        if candidate.exists():
            return candidate

    name = csv_path.name
    if name.startswith("ATL24_"):
        parts = name.split("_")
        if len(parts) >= 3:
            prefix = f"ATL24_{parts[1]}_{parts[2]}"
            matches = sorted(ATL24_H5_DIR.glob(prefix + "*.h5"))
            if matches:
                return matches[0]
    return None


def _backup_once(csv_path: Path) -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / csv_path.name
    if not backup_path.exists():
        backup_path.write_bytes(csv_path.read_bytes())


def _fill_one_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path, low_memory=False)
    h5_path = _find_h5_for_csv(df, csv_path)
    if h5_path is None:
        return _result(csv_path, "failed", "matching ATL24 H5 not found", len(df))

    if "Beam" not in df.columns:
        return _result(csv_path, "failed", "missing Beam column", len(df))
    if "index_ph" not in df.columns:
        return _result(csv_path, "failed", "missing index_ph column", len(df))

    beams = [str(b) for b in df["Beam"].dropna().unique()]
    if len(beams) != 1:
        return _result(csv_path, "failed", f"expected one beam, found {beams}", len(df))
    beam = beams[0]

    h5_df = _read_beam_table(h5_path, beam)
    work = df.copy()
    work["index_ph"] = pd.to_numeric(work["index_ph"], errors="coerce").astype("Int64")
    h5_df["index_ph"] = pd.to_numeric(h5_df["index_ph"], errors="coerce").astype("Int64")
    merged = work[["index_ph"]].merge(h5_df, on="index_ph", how="left", sort=False)

    filled = []
    for h5_name, csv_col in H5_TO_CSV_COLUMNS.items():
        if h5_name in merged.columns:
            work[csv_col] = merged[h5_name].to_numpy()
            filled.append(csv_col)

    for csv_col, h5_name in ALIASES.items():
        if h5_name in merged.columns:
            work[csv_col] = merged[h5_name].to_numpy()
            filled.append(csv_col)

    for col in NOT_IN_ATL24_H5:
        if col not in work.columns:
            work[col] = np.nan

    _backup_once(csv_path)
    work.to_csv(csv_path, index=False, encoding="utf-8-sig")
    matched = int(merged["lat_ph"].notna().sum()) if "lat_ph" in merged.columns else 0
    return _result(
        csv_path,
        "success",
        f"matched_h5={h5_path.name}; beam={beam}; matched_rows={matched}",
        len(work),
        filled_columns=",".join(sorted(set(filled))),
    )


def _result(csv_path, status, message, rows, filled_columns=""):
    return {
        "csv_name": csv_path.name,
        "status": status,
        "message": message,
        "rows": rows,
        "filled_columns": filled_columns,
        "missing_from_h5": ",".join(NOT_IN_ATL24_H5),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill ATL24 CSV columns from downloaded ATL24 H5 files in-place."
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(
        p for p in ATL24_CSV_DIR.glob("*.csv")
        if p.name != REPORT_PATH.name and not p.name.endswith("summary.csv")
    )
    if args.limit is not None:
        files = files[:args.limit]

    rows = []
    for i, csv_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {csv_path.name}", flush=True)
        result = _fill_one_csv(csv_path)
        rows.append(result)
        print(f"  {result['status']}: {result['message']}", flush=True)

    pd.DataFrame(rows).to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    ok = sum(r["status"] == "success" for r in rows)
    print(f"Done. Successful files: {ok}/{len(rows)}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
