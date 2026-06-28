from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(r"H:\ATL24\PhisicKPConvNet\validation_area\data\Florida Bay\Predictions_1")
ATL24_DIR = BASE_DIR / "ATL24_CSV"
REPORT_PATH = ATL24_DIR / "atl24_to_atl03_strict_match_dry_run.csv"

ATL24_SKIP_NAMES = {
    "atl24_export_summary.csv",
    "atl24_fill_missing_columns_report.csv",
    "atl24_to_atl03_strict_match_dry_run.csv",
}

ATL24_LABEL_MAP = {0: 0, 40: 3, 41: 1}


def _matching_atl03_csv(atl24_csv: Path) -> Path:
    name = atl24_csv.name
    if not name.startswith("ATL24_"):
        return BASE_DIR / "__missing__"
    name = "ATL03_" + name[len("ATL24_"):]
    if name.endswith("_labeled.csv"):
        name = name[:-len("_labeled.csv")] + "_compared.csv"
    return BASE_DIR / name


def _read_csv(path: Path, usecols):
    return pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)


def _dup_count(df: pd.DataFrame, cols) -> int:
    if not all(c in df.columns for c in cols):
        return -1
    return int(df.duplicated(cols, keep=False).sum())


def _match_rate_by_single_int_key(src, dst, src_col, dst_values, dst_name):
    if src_col not in src.columns:
        return None
    src_key = pd.to_numeric(src[src_col], errors="coerce")
    valid = src_key.notna()
    dst_set = set(int(x) for x in dst_values)
    matched = src_key[valid].astype(np.int64).isin(dst_set)
    return {
        f"{dst_name}_valid": int(valid.sum()),
        f"{dst_name}_matched": int(matched.sum()),
        f"{dst_name}_match_rate": float(matched.mean()) if len(matched) else 0.0,
    }


def _coordinate_key(df, lat_col, lon_col, h_col, lat_dec=10, lon_dec=10, h_dec=4):
    return (
        pd.to_numeric(df[lat_col], errors="coerce").round(lat_dec).astype("string")
        + "|"
        + pd.to_numeric(df[lon_col], errors="coerce").round(lon_dec).astype("string")
        + "|"
        + pd.to_numeric(df[h_col], errors="coerce").round(h_dec).astype("string")
    )


def _coord_match(src, dst, src_cols, dst_cols, lat_dec=10, lon_dec=10, h_dec=4):
    if not all(c in src.columns for c in src_cols) or not all(c in dst.columns for c in dst_cols):
        return {
            "coord_key_available": False,
            "coord_match_rate": 0.0,
            "coord_matched": 0,
            "coord_dup_src_rows": -1,
            "coord_dup_dst_rows": -1,
        }
    src_key = _coordinate_key(src, *src_cols, lat_dec=lat_dec, lon_dec=lon_dec, h_dec=h_dec)
    dst_key = _coordinate_key(dst, *dst_cols, lat_dec=lat_dec, lon_dec=lon_dec, h_dec=h_dec)
    src_valid = ~src_key.str.contains("<NA>", regex=False, na=True)
    dst_valid = ~dst_key.str.contains("<NA>", regex=False, na=True)
    dst_set = set(dst_key[dst_valid].tolist())
    matched = src_key[src_valid].isin(dst_set)
    return {
        "coord_key_available": True,
        "coord_precision": f"lat{lat_dec}_lon{lon_dec}_h{h_dec}",
        "coord_valid_src": int(src_valid.sum()),
        "coord_valid_dst": int(dst_valid.sum()),
        "coord_matched": int(matched.sum()),
        "coord_match_rate": float(matched.mean()) if len(matched) else 0.0,
        "coord_dup_src_rows": int(src_key[src_valid].duplicated(keep=False).sum()),
        "coord_dup_dst_rows": int(dst_key[dst_valid].duplicated(keep=False).sum()),
    }


def _label_counts(df):
    if "Label" not in df.columns:
        return ""
    counts = df["Label"].value_counts(dropna=False).sort_index()
    return ";".join(f"{k}:{v}" for k, v in counts.items())


def diagnose_one(atl24_csv: Path):
    target_csv = _matching_atl03_csv(atl24_csv)
    row = {
        "atl24_csv": atl24_csv.name,
        "atl03_csv": target_csv.name,
        "target_exists": target_csv.exists(),
    }
    if not target_csv.exists():
        row["status"] = "failed_missing_target"
        return row

    atl24_cols = {
        "Latitude", "Longitude", "Geoid_Corrected_Ortho_Height", "Label",
        "index_ph", "Ph_Index1", "Ph_Index2", "Beam",
        "lat_ph", "lon_ph", "h_ph", "ATL24_index_seg",
    }
    atl03_cols = {
        "Latitude", "Longitude", "Geoid_Corrected_Ortho_Height",
        "lat_ph", "lon_ph", "h_ph",
        "WGS84_Ellipsoid_Height",
        "segment_id", "ph_id_pulse", "ph_id_channel", "ph_id_count",
    }
    src = _read_csv(atl24_csv, atl24_cols)
    dst = _read_csv(target_csv, atl03_cols)

    row.update({
        "status": "ok",
        "atl24_rows": len(src),
        "atl03_rows": len(dst),
        "atl24_label_counts": _label_counts(src),
        "has_atl24_photon_id_fields": all(c in src.columns for c in ["segment_id", "ph_id_pulse", "ph_id_channel", "ph_id_count"]),
        "has_atl03_photon_id_fields": all(c in dst.columns for c in ["segment_id", "ph_id_pulse", "ph_id_channel", "ph_id_count"]),
    })

    for src_col in ["index_ph", "Ph_Index1", "Ph_Index2"]:
        if src_col in src.columns:
            row.update(_match_rate_by_single_int_key(src, dst, src_col, range(len(dst)), f"{src_col}_to_zero_based_row"))
            row.update(_match_rate_by_single_int_key(src, dst, src_col, range(1, len(dst) + 1), f"{src_col}_to_one_based_row"))

    if "ATL24_index_seg" in src.columns and "segment_id" in dst.columns:
        for offset in (-1, 0, 1):
            src_seg = pd.to_numeric(src["ATL24_index_seg"], errors="coerce") + offset
            dst_seg_set = set(pd.to_numeric(dst["segment_id"], errors="coerce").dropna().astype(np.int64).tolist())
            valid = src_seg.notna()
            matched = src_seg[valid].astype(np.int64).isin(dst_seg_set)
            prefix = f"atl24_index_seg_plus_{offset}_to_segment_id"
            row[f"{prefix}_match_rate"] = float(matched.mean()) if len(matched) else 0.0
            row[f"{prefix}_matched"] = int(matched.sum())
            row[f"{prefix}_duplicate_src_rows"] = int(src_seg[valid].duplicated(keep=False).sum())

    # Exact/quantized coordinate-height strict keys. These are equality joins
    # after deterministic rounding, not nearest-neighbour matching.
    for decs in [(12, 12, 6), (10, 10, 4), (8, 8, 3), (7, 7, 2)]:
        cm = _coord_match(
            src,
            dst,
            ("Latitude", "Longitude", "Geoid_Corrected_Ortho_Height"),
            ("Latitude", "Longitude", "Geoid_Corrected_Ortho_Height"),
            lat_dec=decs[0],
            lon_dec=decs[1],
            h_dec=decs[2],
        )
        prefix = f"coord_{decs[0]}_{decs[1]}_{decs[2]}_"
        row.update({prefix + k: v for k, v in cm.items()})

    for decs in [(12, 12, 6), (10, 10, 4), (8, 8, 3), (7, 7, 2)]:
        cm = _coord_match(
            src,
            dst,
            ("lat_ph", "lon_ph", "h_ph"),
            ("lat_ph", "lon_ph", "h_ph"),
            lat_dec=decs[0],
            lon_dec=decs[1],
            h_dec=decs[2],
        )
        prefix = f"rawph_{decs[0]}_{decs[1]}_{decs[2]}_"
        row.update({prefix + k: v for k, v in cm.items()})

    for decs in [(12, 12, 6), (10, 10, 4), (8, 8, 3), (7, 7, 2)]:
        cm = _coord_match(
            src,
            dst,
            ("Latitude", "Longitude", "WGS84_Ellipsoid_Height"),
            ("Latitude", "Longitude", "WGS84_Ellipsoid_Height"),
            lat_dec=decs[0],
            lon_dec=decs[1],
            h_dec=decs[2],
        )
        prefix = f"wgs84_{decs[0]}_{decs[1]}_{decs[2]}_"
        row.update({prefix + k: v for k, v in cm.items()})

    return row


def main():
    files = sorted(
        p for p in ATL24_DIR.glob("*.csv")
        if p.name not in ATL24_SKIP_NAMES and not p.name.endswith("summary.csv")
    )
    rows = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}", flush=True)
        row = diagnose_one(path)
        rows.append(row)
        print(f"  {row.get('status')} target={row.get('target_exists')}", flush=True)

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    print(f"Report: {REPORT_PATH}")
    if not report.empty:
        cols = [c for c in report.columns if c.endswith("match_rate")]
        print(report[["atl24_csv", "status"] + cols].to_string(index=False))


if __name__ == "__main__":
    main()
