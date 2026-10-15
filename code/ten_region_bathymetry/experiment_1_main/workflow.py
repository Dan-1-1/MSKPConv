"""Core implementation for the ten-region bathymetry experiment.

The module is deliberately organized as small, stage-oriented functions.  Each
stage reads only its declared inputs and writes to a run-specific output
directory, so a failed stage can be resumed without changing source products.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
import yaml
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

FEATURE_FILES = {
    "B02": "B02.tif", "B03": "B03.tif", "B04": "B04.tif", "B08": "B08.tif",
    "NDWI": "NDWI.tif", "Blue_Green_LogRatio": "Blue_Green_LogRatio.tif",
    "Blue_Red_LogRatio": "Blue_Red_LogRatio.tif", "Green_Red_LogRatio": "Green_Red_LogRatio.tif",
}
BASE_METHODS = ("PointNetPP", "PointNeXt", "KPConv", "DGCNN", "ATL24", "Song2024")
METHODS = BASE_METHODS + ("Proposed",)
METHOD_SUFFIXES = {"ATL24": "ATL24_Label", "Song2024": "Song2024_Project_Class"}
SCENE_RE = re.compile(r"_(\d{8}T\d{6})_")
TRACK_RE = re.compile(r"ATL03_(\d{14})_(\d{8})_\d{3}_\d{2}_(gt[123][lr])", re.I)
REQUIRED_SCENE_FILES = tuple(FEATURE_FILES.values()) + ("cloud_mask.tif", "metadata.json")
STATION_META = {
    "8723970": (24.7117, -81.1050, "Vaca Key"), "8723214": (25.7314, -80.1618, "Virginia Key"),
    "8724580": (24.5557, -81.8079, "Key West"), "9751381": (18.3117, -64.7233, "Lameshur Bay"),
    "9751639": (18.3419, -64.9300, "Charlotte Amalie"), "9752235": (18.3033, -65.3000, "Culebra"),
    "9751364": (17.7477, -64.7049, "Christiansted"), "1770000": (-14.2756, -170.6895, "Pago Pago"),
    "1630000": (13.4410, 144.6560, "Apra Harbor"), "1631428": (13.2730, 144.6940, "Pago Bay"),
    "1615680": (20.8947, -156.4671, "Kahului"), "1617433": (20.0389, -155.8278, "Kawaihae"),
    "1612480": (21.4330, -157.7900, "Mokuoloe"), "1612340": (21.2983, -157.8667, "Honolulu"),
    "1612401": (21.3100, -158.0700, "Pearl Harbor"),
}


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    csv_root: Path
    s2_root: Path
    cudem: Path
    proposed_suffix: str
    local_crs: str
    representative_scene: str
    tide_stations: tuple[str, ...]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def regions_from_config(cfg: dict[str, Any]) -> list[Region]:
    root = Path(cfg["data_root"])
    return [Region(x["id"], x["name"], root / x["csv_root"], root / x["s2_root"],
                   root / x["cudem"], x["proposed_suffix"], x["local_crs"],
                   x["representative_scene"], tuple(str(s) for s in x["tide_stations"]))
            for x in cfg["regions"]]


def run_dir(cfg: dict[str, Any], run_id: str) -> Path:
    p = Path(cfg["output_root"]) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_run_dir(cfg: dict[str, Any], run_id: str) -> Path:
    """Return the run-specific checkpoint root, optionally on another drive."""
    root = Path(cfg.get("model_output_root", cfg["output_root"])) / run_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def finite_rows(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Keep only rows finite across all model inputs and target values."""
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    keep = np.isfinite(numeric.to_numpy(dtype="float64")).all(axis=1)
    clean = frame.loc[keep].copy()
    clean.loc[:, columns] = numeric.loc[keep]
    return clean


def depth_domain_mask(values: Any, cfg: dict[str, Any], *, prefix: str = "evaluation") -> np.ndarray:
    """Return the configured full-scene depth-domain mask.

    The upper bound is intentionally optional.  A null ``*_max_m`` means that
    deep-water observations are retained rather than silently discarded.
    """
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    keep = np.isfinite(numeric)
    minimum = cfg.get(f"{prefix}_min_m", 0.0)
    if minimum is not None:
        keep &= numeric >= float(minimum)
    maximum = cfg.get(f"{prefix}_max_m")
    if maximum is not None:
        keep &= numeric <= float(maximum)
    return keep


def apply_prediction_domain(pred: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Mask invalid predictions while leaving the upper range unrestricted."""
    pred = np.asarray(pred, dtype="float32")
    pred[pred < 0] = np.nan
    maximum = cfg.get("prediction_depth_max_m")
    if maximum is not None:
        pred[pred > float(maximum)] = np.nan
    return pred


def method_suffix(region: Region, method: str) -> str:
    if method in METHOD_SUFFIXES:
        return METHOD_SUFFIXES[method]
    return region.proposed_suffix if method == "Proposed" else method


def method_columns(region: Region, method: str) -> dict[str, str]:
    suffix = method_suffix(region, method)
    label = suffix if method in METHOD_SUFFIXES else f"Pred_{suffix}"
    return {"label": label, "depth": f"Depth_{suffix}",
            "lat": f"Corrected_Latitude_{suffix}", "lon": f"Corrected_Longitude_{suffix}",
            "corr_method": f"Depth_Correction_Method_{suffix}"}


def csv_files(region: Region) -> list[Path]:
    # Deliberately do not recurse: backups and historical copies are excluded.
    return sorted(region.csv_root.glob("*.csv"))


def _region_key(value: Any) -> str:
    """Normalize manifest region labels without relying on punctuation."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def load_track_selection(cfg: dict[str, Any], regions: list[Region]) -> dict[str, list[dict[str, Any]]]:
    """Load the optional selected-track manifest used for a new matching run.

    The manifest is deliberately an input selector, not a second source of
    photon values.  Each selected row is resolved to its original prediction
    CSV and carries the audit metadata (depth formula and vertical offset)
    alongside it.  An absent manifest preserves the historical all-CSV mode.
    """
    raw_path = cfg.get("matching_track_selection_csv")
    if not raw_path:
        return {}
    manifest_path = Path(str(raw_path))
    if not manifest_path.exists():
        raise FileNotFoundError(f"matching_track_selection_csv does not exist: {manifest_path}")
    frame = pd.read_csv(manifest_path, low_memory=False)
    required = {"region", "csv_file", "selected"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"track-selection manifest is missing columns: {missing}")
    selected_mask = frame["selected"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    frame = frame.loc[selected_mask].copy()
    aliases: dict[str, str] = {}
    for region in regions:
        aliases[_region_key(region.id)] = region.id
        aliases[_region_key(region.name)] = region.id
    # The selection table uses the shorter label "Cocos", while the pipeline
    # region is named "Cocos/Guam".
    aliases[_region_key("Cocos")] = "cocos_guam"
    selection: dict[str, list[dict[str, Any]]] = {region.id: [] for region in regions}
    missing_files: list[str] = []
    for _, row in frame.iterrows():
        region_id = aliases.get(_region_key(row["region"]))
        if region_id is None:
            raise ValueError(f"Unknown region in track-selection manifest: {row['region']!r}")
        csv_path_value = str(row.get("csv_path", "")).strip()
        if csv_path_value and csv_path_value.lower() != "nan":
            csv_path = Path(csv_path_value)
        else:
            configured = next(region for region in regions if region.id == region_id)
            csv_path = configured.csv_root / str(row["csv_file"])
        if not csv_path.is_absolute():
            csv_path = Path(str(cfg.get("workspace_root", "."))) / csv_path
        csv_path = csv_path.resolve()
        if not csv_path.is_file():
            missing_files.append(str(csv_path))
            continue
        offset = pd.to_numeric(pd.Series([row.get("tutuila_depth_offset_m", 0.0)]), errors="coerce").iloc[0]
        selection[region_id].append({
            "path": str(csv_path),
            "region": str(row["region"]),
            "csv_file": str(row["csv_file"]),
            "depth_column": str(row.get("depth_column", "")),
            "depth_formula": str(row.get("depth_formula", "")),
            "tide_mllw_m": float(pd.to_numeric(pd.Series([row.get("tide_mllw_m")]), errors="coerce").iloc[0])
            if pd.notna(pd.to_numeric(pd.Series([row.get("tide_mllw_m")]), errors="coerce").iloc[0]) else None,
            "tutuila_depth_offset_m": float(offset) if pd.notna(offset) else 0.0,
            "threshold_r2": row.get("threshold_r2"),
            "source_manifest": str(manifest_path.resolve()),
        })
    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(
            f"{len(missing_files)} selected prediction CSV files are missing; first files:\n{preview}"
        )
    return selection


def selected_csv_files(region: Region, selection: dict[str, list[dict[str, Any]]] | None) -> list[Path]:
    if selection is None:
        return csv_files(region)
    return [Path(item["path"]) for item in selection.get(region.id, [])]


def parse_track(path: Path) -> tuple[str, datetime] | tuple[None, None]:
    m = TRACK_RE.search(path.name)
    if not m:
        return None, None
    dt = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    return f"{m.group(1)}_{m.group(2)}_{m.group(3).lower()}", dt


def scene_time(scene_dir: Path) -> datetime | None:
    try:
        meta = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
        candidates = meta.get("source_scenes", []) or meta.get("merged_scenes", [])
        if candidates:
            m = SCENE_RE.search(str(candidates[0]))
            if m:
                return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
        m = re.search(r"(\d{6})", scene_dir.name)
        return datetime.strptime(m.group(1), "%Y%m") if m else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def scene_inventory(region: Region) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    complete, excluded = [], []
    for d in sorted(region.s2_root.glob("S2_Mosaic_*")):
        if not d.is_dir():
            continue
        missing = [f for f in REQUIRED_SCENE_FILES if not (d / f).is_file()]
        t = scene_time(d)
        if t is None:
            missing.append("metadata.source_scenes[0] time")
        item = {"scene_id": d.name, "scene_dir": str(d), "scene_time": t.isoformat() if t else None,
                "missing": missing}
        if missing:
            excluded.append(item)
        else:
            complete.append(item)
    return complete, excluded


def header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def qc(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    regions = regions_from_config(cfg)
    selection = load_track_selection(cfg, regions)
    out = run_dir(cfg, run_id) / "qc"
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"run_id": run_id, "generated_at": datetime.utcnow().isoformat() + "Z", "regions": []}
    total_csv = total_scene = 0
    for r in regions:
        files = selected_csv_files(r, selection if selection else None)
        complete, excluded = scene_inventory(r)
        total_csv += len(files); total_scene += len(complete)
        expected = {"Latitude", "Longitude"}
        fields = {m: method_columns(r, m) for m in METHODS}
        missing_by_file = []
        for p in files:
            try: cols = set(header(p))
            except Exception as exc: missing_by_file.append({"file": p.name, "error": str(exc)}); continue
            req = expected | {c for d in fields.values() for c in d.values()}
            missing = sorted(req - cols)
            if missing: missing_by_file.append({"file": p.name, "missing": missing})
        region_item = {"id": r.id, "name": r.name, "csv_count": len(files),
                       "track_selection_manifest": cfg.get("matching_track_selection_csv"),
                       "complete_scene_count": len(complete), "excluded_scenes": excluded,
                       "field_issues": missing_by_file, "local_crs": r.local_crs}
        result["regions"].append(region_item)
        pd.DataFrame([{"region_id": r.id, "file": p.name, "track_group_id": parse_track(p)[0]}
                      for p in files]).to_csv(out / f"{r.id}_files.csv", index=False)
    any_field_issues = any(x["field_issues"] for x in result["regions"])
    expected_csv = sum(len(selected_csv_files(r, selection if selection else None)) for r in regions)
    result.update({"total_csv": total_csv, "total_complete_scenes": total_scene,
                   "expected_csv": expected_csv, "expected_complete_scenes": total_scene,
                   "status": ("PASS_WITH_WARNINGS" if any_field_issues else "PASS")
                   if total_csv == expected_csv else "WARN",
                   "track_selection_manifest": cfg.get("matching_track_selection_csv"),
                   "selected_track_count": expected_csv if selection else None})
    write_json(out / "qc_summary.json", result)
    print(json.dumps({k: result[k] for k in ("total_csv", "total_complete_scenes", "status")}, ensure_ascii=False))
    return result


class TideProvider:
    """Fetch and cache observed NOAA MLLW values; missing data is never zero-filled."""
    def __init__(self, cache_path: Path):
        self.path = cache_path
        self.data = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    def _save(self):
        write_json(self.path, self.data)

    def _month(self, station: str, dt: datetime) -> pd.DataFrame:
        key = f"{station}_{dt:%Y%m}"
        if key in self.data:
            d = self.data[key]
            return pd.DataFrame({"t": pd.to_datetime(d["t"]), "v": np.asarray(d["v"], dtype=float)})
        first = dt.replace(day=1, hour=0, minute=0, second=0)
        next_month = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
        params = {"product": "water_level", "application": "ten_region_bathymetry", "begin_date": first.strftime("%Y%m%d"),
                  "end_date": (next_month - timedelta(minutes=6)).strftime("%Y%m%d %H:%M"), "datum": "MLLW",
                  "station": station, "time_zone": "gmt", "units": "metric", "format": "json"}
        response = requests.get("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if "data" not in payload or not payload["data"]:
            raise RuntimeError(payload.get("error", f"no NOAA data for {station} {key}"))
        df = pd.DataFrame(payload["data"])
        df["t"] = pd.to_datetime(df["t"], errors="coerce"); df["v"] = pd.to_numeric(df["v"], errors="coerce")
        df = df.dropna(subset=["t", "v"])
        if df.empty: raise RuntimeError(f"empty NOAA data for {station} {key}")
        self.data[key] = {"t": [x.isoformat() for x in df["t"]], "v": df["v"].astype(float).tolist()}
        self._save(); time.sleep(0.05)
        return df

    def value(self, stations: Iterable[str], dt: datetime) -> tuple[float, str]:
        errors = []
        for station in stations:
            try:
                df = self._month(str(station), dt)
                idx = (df["t"] - pd.Timestamp(dt)).abs().idxmin()
                return float(df.loc[idx, "v"]), str(station)
            except Exception as exc:
                errors.append(f"{station}: {exc}")
        raise RuntimeError("NOAA tide unavailable: " + " | ".join(errors))


def nearest_scene(scenes: list[dict[str, Any]], dt: datetime, max_days: int) -> dict[str, Any] | None:
    if not scenes: return None
    rows = [(abs(datetime.fromisoformat(x["scene_time"]) - dt), x) for x in scenes]
    delta, item = min(rows, key=lambda z: z[0])
    return item if delta <= timedelta(days=max_days) else None


def sample_scene(scene_dir: Path, lats: np.ndarray, lons: np.ndarray, features: list[str]) -> pd.DataFrame:
    import rasterio
    with rasterio.open(scene_dir / FEATURE_FILES["B02"]) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows, cols = np.asarray(rows), np.asarray(cols)
        valid = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
    out = {"_source_index": np.flatnonzero(valid), "img_row": rows[valid].astype(int), "img_col": cols[valid].astype(int)}
    if not np.any(valid): return pd.DataFrame(out)
    rr, cc = out["img_row"], out["img_col"]
    for feature in features:
        with rasterio.open(scene_dir / FEATURE_FILES[feature]) as src:
            arr = src.read(1).astype(float); vals = arr[rr, cc]
            if src.nodata is not None: vals[vals == src.nodata] = np.nan
            out[feature] = vals
    with rasterio.open(scene_dir / "cloud_mask.tif") as src: out["cloud_mask"] = src.read(1)[rr, cc]
    frame = pd.DataFrame(out)
    good = frame["cloud_mask"].eq(0)
    good &= frame[features].notna().all(axis=1)
    good &= frame[["B02", "B03", "B04", "B08"]].gt(0).all(axis=1)
    return frame.loc[good].reset_index(drop=True)


def mad_keep(values: np.ndarray) -> np.ndarray:
    med = np.nanmedian(values); mad = np.nanmedian(np.abs(values - med))
    return np.abs(values - med) <= (3 * mad if mad > 0 else 0)


def aggregate(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if frame.empty: return frame
    keys = ["track_group_id", "scene_id", "img_row", "img_col"]
    rows = []
    for key, g in frame.groupby(keys, sort=False):
        vals = g["Depth"].to_numpy(float); keep = mad_keep(vals)
        if keep.sum() < 3: continue
        first = g.iloc[0]
        row = {k: v for k, v in zip(keys, key)}
        row.update({"region_id": first["region_id"], "method": first["method"], "scene_time": first["scene_time"],
                    "track_time": first["track_time"], "Latitude": g["Latitude"].mean(), "Longitude": g["Longitude"].mean(),
                    "Depth": float(np.median(vals[keep])), "Photon_Count": int(len(g)),
                    "Photon_Count_Filtered": int(keep.sum()), "Depth_Std": float(np.std(vals[keep], ddof=1))})
        row.update({f: float(first[f]) for f in features})
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_region(
    cfg: dict[str, Any],
    region: Region,
    run_path: Path,
    tide: TideProvider,
    force_methods: set[str] | None = None,
    selection: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    features = list(cfg["features"]); scenes, excluded = scene_inventory(region)
    region_files = selected_csv_files(region, selection)
    selection_by_path = {
        str(Path(item["path"]).resolve()): item
        for item in (selection or {}).get(region.id, [])
    }
    scene_by_id = {x["scene_id"]: x for x in scenes}
    reuse = bool(cfg.get("reuse_prepared_methods", False))
    forced = force_methods or set()
    prepare_methods = [
        m for m in METHODS
        if m in forced or not (reuse and (run_path / "prepared" / region.id / f"{m}.parquet").exists())
    ]
    records: dict[str, list[pd.DataFrame]] = {m: [] for m in prepare_methods}; errors = []
    if not prepare_methods:
        out = {"region_id": region.id, "csv_count": len(region_files), "complete_scene_count": len(scenes),
               "excluded_scene_count": len(excluded), "errors": [], "tide_station_cache": str(tide.path),
               "depth_match_min_m": cfg.get("depth_match_min_m", 0.0),
               "depth_match_max_m": cfg.get("depth_match_max_m"),
               "reused_methods": list(METHODS)}
        for method in METHODS:
            p = run_path / "prepared" / region.id / f"{method}.parquet"
            out[f"{method}_rows"] = int(len(pd.read_parquet(p))) if p.exists() else 0
        write_json(run_path / "prepared" / region.id / "summary.json", out)
        return out
    for idx, path in enumerate(region_files, 1):
        selection_record = selection_by_path.get(str(path.resolve()), {})
        track_id, track_dt = parse_track(path)
        if track_id is None: errors.append({"file": path.name, "error": "unparseable ATL03 filename"}); continue
        scene = nearest_scene(scenes, track_dt, int(cfg["scene_match_days"]))
        if scene is None: errors.append({"file": path.name, "error": "no complete Sentinel-2 scene within 45 days"}); continue
        scene_dt = datetime.fromisoformat(scene["scene_time"])
        # The selected-track table already contains the tide value used by
        # the published correction (Depth_refracted - Tide_MLLW).  Prefer it
        # over a second NOAA lookup so the new run reproduces that correction,
        # including historical Tutuila tracks for which the station API has no
        # archived monthly product.  The fallback retains the legacy scene-
        # time conversion for runs without a selection manifest.
        manifest_tide = selection_record.get("tide_mllw_m")
        if manifest_tide is not None and np.isfinite(float(manifest_tide)):
            tide_is2 = float(manifest_tide)
            tide_s2 = 0.0
        else:
            try:
                tide_is2, st1 = tide.value(region.tide_stations, track_dt)
                tide_s2, st2 = tide.value(region.tide_stations, scene_dt)
            except Exception as exc:
                errors.append({"file": path.name, "scene_id": scene["scene_id"], "error": str(exc)}); continue
        needed = {"Latitude", "Longitude"}
        for m in prepare_methods: needed.update(method_columns(region, m).values())
        try: frame = pd.read_csv(path, usecols=lambda c: c in needed, low_memory=False)
        except Exception as exc: errors.append({"file": path.name, "error": str(exc)}); continue
        for method in prepare_methods:
            c = method_columns(region, method)
            if not all(x in frame for x in c.values()): continue
            x = frame.loc[pd.to_numeric(frame[c["label"]], errors="coerce").eq(3)].copy()
            for col in (c["lat"], c["lon"], c["depth"]): x[col] = pd.to_numeric(x[col], errors="coerce")
            x = x.dropna(subset=[c["lat"], c["lon"], c["depth"]])
            x = x[depth_domain_mask(x[c["depth"]], cfg, prefix="depth_match")]
            if x.empty: continue
            sampled = sample_scene(Path(scene["scene_dir"]), x[c["lat"]].to_numpy(), x[c["lon"]].to_numpy(), features)
            if sampled.empty: continue
            # Preserve exact row correspondence after cloud/nodata filtering.
            x = x.iloc[sampled.pop("_source_index").to_numpy(dtype=int)].reset_index(drop=True)
            # Source depths are referenced to the photon acquisition surface.
            # Convert to the matched Sentinel-2 scene time, then apply any
            # region/track vertical datum correction recorded in the manifest
            # (Tutuila currently uses +2.5 m).
            depth_offset = float(selection_record.get("tutuila_depth_offset_m", 0.0) or 0.0)
            sampled["Depth"] = x[c["depth"]].to_numpy(float) + tide_s2 - tide_is2 + depth_offset
            sampled["Latitude"] = x[c["lat"]].to_numpy(float); sampled["Longitude"] = x[c["lon"]].to_numpy(float)
            sampled["track_group_id"] = track_id; sampled["scene_id"] = scene["scene_id"]
            sampled["scene_time"] = scene_dt.isoformat(); sampled["track_time"] = track_dt.isoformat()
            sampled["region_id"] = region.id; sampled["method"] = method
            records[method].append(aggregate(sampled, features))
        if idx % 25 == 0: print(f"prepare {region.id}: {idx}/{len(region_files)}", flush=True)
    out = {"region_id": region.id, "csv_count": len(region_files), "complete_scene_count": len(scenes),
           "excluded_scene_count": len(excluded), "errors": errors, "tide_station_cache": str(tide.path),
           "depth_match_min_m": cfg.get("depth_match_min_m", 0.0),
           "depth_match_max_m": cfg.get("depth_match_max_m")}
    out["track_selection_manifest"] = cfg.get("matching_track_selection_csv")
    out["selected_track_count"] = len(region_files) if selection is not None else None
    for method, frames in records.items():
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        dest = run_path / "prepared" / region.id / f"{method}.parquet"; dest.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(dest, index=False)
        out[f"{method}_rows"] = int(len(data))
    for method in METHODS:
        if method not in records:
            p = run_path / "prepared" / region.id / f"{method}.parquet"
            out[f"{method}_rows"] = int(len(pd.read_parquet(p))) if p.exists() else 0
    out["reused_methods"] = [m for m in METHODS if m not in records]
    write_json(run_path / "prepared" / region.id / "summary.json", out)
    return out


def prepare(
    cfg: dict[str, Any],
    run_id: str,
    region_ids: set[str] | None = None,
    force_methods: set[str] | None = None,
) -> dict[str, Any]:
    path = run_dir(cfg, run_id); tide = TideProvider(path / "prepared" / "noaa_tide_cache.json")
    regions = regions_from_config(cfg)
    selection = load_track_selection(cfg, regions)
    known_regions = {r.id for r in regions}
    unknown_regions = (region_ids or set()) - known_regions
    unknown_methods = (force_methods or set()) - set(METHODS)
    if unknown_regions:
        raise ValueError(f"Unknown region IDs: {sorted(unknown_regions)}")
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")
    selected = [r for r in regions if region_ids is None or r.id in region_ids]
    summary_path = path / "prepared" / "summary.json"
    if summary_path.exists():
        try:
            result = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = {}
    else:
        result = {}
    if selection:
        write_json(path / "prepared" / "track_selection_manifest.json", {
            "source_manifest": cfg.get("matching_track_selection_csv"),
            "selected_track_count": int(sum(len(v) for v in selection.values())),
            "regions": selection,
        })
    for region in selected:
        result[region.id] = prepare_region(cfg, region, path, tide, force_methods, selection or None)
    write_json(path / "prepared" / "summary.json", result); return result


def stable_order(group: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16)


def assign_validation_groups(frame: pd.DataFrame, split_col: str,
                             validation_groups: Iterable[str]) -> pd.DataFrame:
    """Mark validation rows without restoring spatially excluded samples."""
    mask = frame[split_col].eq("train") & frame["track_group_id"].isin(validation_groups)
    frame.loc[mask, split_col] = "validation"
    return frame


def split(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    path = run_dir(cfg, run_id); result = {}
    for r in regions_from_config(cfg):
        method_data = {}
        all_groups = {}
        empty_methods = []
        for m in METHODS:
            p = path / "prepared" / r.id / f"{m}.parquet"
            if not p.exists(): continue
            d = pd.read_parquet(p)
            # A completed matcher may legitimately produce no usable rows for
            # a region. Treat that as an auditable no-data outcome instead of
            # attempting to re-run matching or failing on a schema-less empty
            # Parquet file.
            if d.empty or "track_group_id" not in d.columns:
                empty_methods.append(m)
                continue
            method_data[m] = d
            for g, n in d.groupby("track_group_id").size().items(): all_groups[g] = all_groups.get(g, 0) + int(n)
        groups = sorted(all_groups, key=lambda g: (-all_groups[g], stable_order(g, int(cfg["seed"]))))
        fold_sizes = [0] * int(cfg["n_folds"]); fold_map = {}
        for g in groups:
            f = int(np.argmin(fold_sizes)); fold_map[g] = f; fold_sizes[f] += all_groups[g]
        # Project and assign folds before processing individual methods.  This
        # lets every method use the Proposed test coordinates as the common
        # spatial exclusion anchor while retaining its own held-out rows for
        # supplementary ICESat-2 diagnostics.
        projected = {}
        transformer = Transformer.from_crs("EPSG:4326", r.local_crs, always_xy=True)
        for m, d in method_data.items():
            d = d.copy(); d["fold"] = d["track_group_id"].map(fold_map).astype(int)
            d["x"], d["y"] = transformer.transform(d["Longitude"].to_numpy(), d["Latitude"].to_numpy())
            projected[m] = d
        anchor_method = str(cfg.get("shared_test_anchor_method", "Proposed"))
        anchor_data = projected.get(anchor_method)
        if anchor_data is None and projected:
            anchor_method, anchor_data = next(iter(projected.items()))
        region_result = {"fold_sizes": fold_sizes, "methods": {}, "empty_methods": empty_methods,
                         "shared_test_anchor_method": anchor_method}
        for m, d in projected.items():
            for f in range(int(cfg["n_folds"])):
                col = f"split_fold_{f}"; d[col] = "train"
                test = d[d.fold == f]; candidate = d[d.fold != f]
                anchor_test = anchor_data[anchor_data.fold == f] if anchor_data is not None else test
                if candidate.empty or (test.empty and anchor_test.empty):
                    continue
                # Buffer against both the shared Proposed anchor and the
                # method's own held-out positions.  The former guarantees a
                # leakage-free common CUDEM evaluation; the latter preserves
                # strict isolation for method-specific ICESat-2 diagnostics.
                reference_parts = [x[["x", "y"]] for x in (anchor_test, test) if not x.empty]
                reference_xy = pd.concat(reference_parts, ignore_index=True).drop_duplicates()
                tree = cKDTree(reference_xy.to_numpy()); dist, _ = tree.query(candidate[["x", "y"]].to_numpy(), k=1)
                d.loc[candidate.index[dist < float(cfg["buffer_m"])], col] = "excluded_spatial_buffer"
                d.loc[d.index[d.fold == f], col] = "test"
                remaining = d[(d.fold != f) & d[col].eq("train")]
                # Select complete tracks, but guarantee a non-trivial early-
                # stopping set. A hash-only 10% rule can produce a validation
                # set of one tiny track, which makes best_iteration=0 a noisy
                # artefact rather than evidence that one tree is optimal.
                ordered_groups = sorted(remaining.track_group_id.unique(),
                                        key=lambda g: stable_order(g, int(cfg["seed"]) + f))
                desired_val_rows = max(int(np.ceil(len(remaining) * float(cfg["early_stop_fraction"]))),
                                       int(cfg.get("min_validation_rows", 100)))
                validation_groups = []
                val_rows = 0
                for g in ordered_groups:
                    validation_groups.append(g)
                    val_rows += int((remaining.track_group_id == g).sum())
                    if val_rows >= desired_val_rows:
                        break
                # Never resurrect rows already excluded by the shared test
                # buffer.  The previous whole-track assignment changed such
                # rows back to validation and allowed near-test observations
                # to influence early stopping and convex-blend weights.
                assign_validation_groups(d, col, validation_groups)
                if validation_groups:
                    val = d[d[col].eq("validation")]
                    tree2 = cKDTree(val[["x", "y"]].to_numpy()); rest = d[d[col].eq("train") & (d.fold != f)]
                    dist2, _ = tree2.query(rest[["x", "y"]].to_numpy(), k=1)
                    d.loc[rest.index[dist2 < float(cfg["buffer_m"])], col] = "excluded_spatial_buffer"
            dest = path / "splits" / r.id / f"{m}.parquet"; dest.parent.mkdir(parents=True, exist_ok=True); d.to_parquet(dest, index=False)
            region_result["methods"][m] = {str(f): {k: int(v) for k, v in d[f"split_fold_{f}"].value_counts().items()} for f in range(int(cfg["n_folds"]))}
        write_json(path / "splits" / r.id / "summary.json", region_result); result[r.id] = region_result
    write_json(path / "splits" / "summary.json", result); return result


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if len(y) == 0: return {k: float("nan") for k in ("n", "rmse", "mae", "median_ae", "bias", "r2", "pearson_r2")}
    corr = np.corrcoef(y, pred)[0, 1] if len(y) > 1 else np.nan
    return {"n": int(len(y)), "rmse": float(mean_squared_error(y, pred) ** 0.5),
            "mae": float(mean_absolute_error(y, pred)), "median_ae": float(median_absolute_error(y, pred)),
            "bias": float(np.mean(pred - y)), "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
            "pearson_r2": float(corr ** 2) if np.isfinite(corr) else np.nan}


def cudem_rasters(region: Region) -> list[Path]:
    """Return configured CUDEM rasters, supporting single files or tile folders."""
    p = region.cudem
    if p.is_file(): return [p]
    if p.is_dir():
        return sorted(x for x in p.rglob("*.tif") if x.is_file() and "land_mask" not in x.name.lower())
    return []


def sample_cudem_depth(region: Region, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Sample CUDEM at lon/lat and convert the dominant elevation sign to depth."""
    import rasterio
    from pyproj import Transformer
    out = np.full(len(lon), np.nan, dtype="float64")
    for raster in cudem_rasters(region):
        try:
            with rasterio.open(raster) as src:
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True) if src.crs else None
                xs, ys = transformer.transform(lon, lat) if transformer else (lon, lat)
                inside = (xs >= src.bounds.left) & (xs <= src.bounds.right) & (ys >= src.bounds.bottom) & (ys <= src.bounds.top) & ~np.isfinite(out)
                if not inside.any(): continue
                vals = np.fromiter((v[0] for v in src.sample(list(zip(xs[inside], ys[inside])))), dtype="float64", count=int(inside.sum()))
                if src.nodata is not None: vals[vals == src.nodata] = np.nan
                out[np.where(inside)[0]] = vals
        except Exception:
            continue
    finite = np.isfinite(out)
    if finite.any() and np.nanmedian(out[finite]) < 0: out[finite] = -out[finite]
    return out


def train(cfg: dict[str, Any], run_id: str, max_tasks: int | None = None, folds: list[int] | None = None) -> dict[str, Any]:
    from catboost import CatBoostRegressor
    path = run_dir(cfg, run_id); features = list(cfg["features"]); cb = dict(cfg["catboost"]); results = []
    if str(cfg.get("training_mode", "regional")).lower() == "pooled":
        return train_pooled(cfg, run_id, max_tasks, folds)
    task_count = 0
    for r in regions_from_config(cfg):
        for m in METHODS:
            p = path / "splits" / r.id / f"{m}.parquet"
            if not p.exists(): continue
            d = pd.read_parquet(p)
            for f in range(int(cfg["n_folds"])):
                if folds is not None and f not in folds: continue
                split_col = f"split_fold_{f}"
                train_df = d[(d.fold != f) & d[split_col].eq("train")]; val_df = d[(d.fold != f) & d[split_col].eq("validation")]; test_df = d[d[split_col].eq("test")]
                min_train = int(cfg.get("min_train_rows", 500)); min_val = int(cfg.get("min_validation_rows", 100))
                if len(train_df) < min_train or len(val_df) < min_val or test_df.empty:
                    results.append({"region_id": r.id, "method": m, "fold": f, "status": "INSUFFICIENT_DATA"}); continue
                model = CatBoostRegressor(**cb, verbose=False, eval_metric="RMSE", od_type="Iter")
                model.fit(train_df[features], train_df.Depth, eval_set=(val_df[features], val_df.Depth), use_best_model=True)
                pred = model.predict(test_df[features]); mm = metrics(test_df.Depth.to_numpy(), pred)
                model_path = path / "models" / r.id / m / f"fold_{f}.cbm"; model_path.parent.mkdir(parents=True, exist_ok=True); model.save_model(model_path)
                pred_path = path / "predictions" / r.id / m / f"fold_{f}.parquet"; pred_path.parent.mkdir(parents=True, exist_ok=True)
                test_out = test_df[["region_id", "method", "fold", "track_group_id", "scene_id", "img_row", "img_col", "Latitude", "Longitude", "Depth"]].copy(); test_out["Prediction"] = pred; test_out.to_parquet(pred_path, index=False)
                results.append({"region_id": r.id, "method": m, "fold": f, "status": "OK", "best_iteration": int(model.get_best_iteration()), **mm})
                task_count += 1; print(f"train {task_count}: {r.id}/{m}/fold{f} n={len(test_df)}", flush=True)
                if max_tasks is not None and task_count >= max_tasks: break
            if max_tasks is not None and task_count >= max_tasks: break
            if max_tasks is None and folds is None:
                good_iters = [x["best_iteration"] for x in results if x.get("region_id") == r.id and x.get("method") == m and x.get("status") == "OK"]
                if good_iters:
                    # CatBoost reports the best iteration as zero-based.  A
                    # one-iteration model is valid; zero iterations is not.
                    full_cb = dict(cb); full_cb["iterations"] = max(1, int(np.median(good_iters)) + 1); full_cb.pop("early_stopping_rounds", None)
                    full_model = CatBoostRegressor(**full_cb, verbose=False, eval_metric="RMSE", od_type="Iter")
                    full_data = d.dropna(subset=features + ["Depth"])
                    full_model.fit(full_data[features], full_data.Depth)
                    full_path = path / "models" / r.id / m / "full.cbm"; full_model.save_model(full_path)
                    results.append({"region_id": r.id, "method": m, "fold": "full", "status": "OK",
                                    "best_iteration": int(np.median(good_iters)), "n": int(len(full_data))})
        if max_tasks is not None and task_count >= max_tasks: break
    df = pd.DataFrame(results); df.to_csv(path / "train_metrics.csv", index=False); write_json(path / "train_summary.json", {"tasks": task_count, "rows": len(df)})
    return {"tasks": task_count, "metrics": df.to_dict("records")}


def _fit_convex_blend_fold(cfg: dict[str, Any], train_df: pd.DataFrame, val_df: pd.DataFrame,
                           test_df: pd.DataFrame, model_root: Path, fold: int,
                           seed: int, save_models: bool = True) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the five base regressors and a validation-constrained convex blend."""
    from joblib import dump
    from experiment_2_model_selection.workflow import _fit_convex_weights, _models, extract_sentinel2_patches

    features = list(cfg["features"])
    base_names = [str(x) for x in cfg["convex_blend"]["base_regressors"]]
    models = _models(cfg, int(seed))
    validation_predictions, test_predictions, model_info = [], [], {}
    cnn_arrays = None
    if "LightweightCNN" in base_names:
        merged = pd.concat([train_df, val_df, test_df], ignore_index=True)
        patches = extract_sentinel2_patches(merged, cfg, int(cfg["lightweight_cnn"].get("image_patch_size", 11)))
        n_train, n_val = len(train_df), len(val_df)
        cnn_arrays = (patches[:n_train], patches[n_train:n_train + n_val], patches[n_train + n_val:])
    for name in base_names:
        model = models[name]
        if name == "LightweightCNN":
            x_train, x_val, x_test = cnn_arrays
            model.fit(x_train, train_df.Depth.to_numpy(float), eval_set=(x_val, val_df.Depth.to_numpy(float)))
            val_pred = model.predict(x_val); test_pred = model.predict(x_test)
            model_info["cnn_best_epoch"] = int(model.best_epoch_)
        elif name == "CatBoost":
            model.fit(train_df[features], train_df.Depth, eval_set=(val_df[features], val_df.Depth), use_best_model=True)
            val_pred = model.predict(val_df[features]); test_pred = model.predict(test_df[features])
            model_info["catboost_best_iteration"] = int(model.get_best_iteration())
        else:
            model.fit(train_df[features], train_df.Depth)
            val_pred = model.predict(val_df[features]); test_pred = model.predict(test_df[features])
        validation_predictions.append(np.asarray(val_pred, dtype="float64"))
        test_predictions.append(np.asarray(test_pred, dtype="float64"))
        if save_models:
            model_path = model_root / name / f"fold_{fold}"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            if name == "CatBoost":
                model.save_model(str(model_path.with_suffix(".cbm")))
            else:
                dump(model, model_path.with_suffix(".joblib"), compress=3)
    validation_matrix = np.column_stack(validation_predictions)
    test_matrix = np.column_stack(test_predictions)
    weights = _fit_convex_weights(val_df.Depth.to_numpy(float), validation_matrix)
    model_info.update({"base_regressors": base_names, "weights": weights.tolist(),
                       "validation_rmse": float(np.sqrt(np.mean((val_df.Depth.to_numpy(float) - validation_matrix @ weights) ** 2)))})
    return test_matrix @ weights, weights, model_info


def train_pooled(cfg: dict[str, Any], run_id: str, max_tasks: int | None = None,
                 folds: list[int] | None = None) -> dict[str, Any]:
    """Train one shared ConvexBlend model per classification method and outer fold."""
    path = run_dir(cfg, run_id); model_path_root = model_run_dir(cfg, run_id); features = list(cfg["features"]); results = []
    datasets = {}
    for method in METHODS:
        parts = []
        for region in regions_from_config(cfg):
            split_path = path / "splits" / region.id / f"{method}.parquet"
            if split_path.exists():
                parts.append(pd.read_parquet(split_path))
        if parts:
            datasets[method] = pd.concat(parts, ignore_index=True)
    task_count = 0
    min_train = int(cfg.get("min_train_rows", 500)); min_val = int(cfg.get("min_validation_rows", 100))
    for method, data in datasets.items():
        for fold in range(int(cfg["n_folds"])):
            if folds is not None and fold not in folds:
                continue
            split_col = f"split_fold_{fold}"
            train_df = finite_rows(data[data[split_col].eq("train") & (data.fold != fold)], features + ["Depth"])
            val_df = finite_rows(data[data[split_col].eq("validation") & (data.fold != fold)], features + ["Depth"])
            test_df = finite_rows(data[data[split_col].eq("test")], features + ["Depth"])
            if len(train_df) < min_train or len(val_df) < min_val or test_df.empty:
                results.append({"scope": "pooled", "method": method, "fold": fold, "regressor": "ConvexBlend",
                                "status": "INSUFFICIENT_DATA", "train_rows": len(train_df),
                                "validation_rows": len(val_df), "test_rows": len(test_df)})
                continue
            pred_path = path / "predictions" / "pooled" / method / f"fold_{fold}.parquet"
            blend_path = model_path_root / "models" / "pooled" / method / "ConvexBlend" / f"fold_{fold}.json"
            if pred_path.exists() and blend_path.exists():
                test_out = pd.read_parquet(pred_path)
                blend_info = json.loads(blend_path.read_text(encoding="utf-8"))
                reused = len(test_out) == len(test_df) and blend_info.get("type") == "validation_fitted_convex_blend"
            else:
                reused = False
            if not reused:
                pred, weights, model_info = _fit_convex_blend_fold(
                    cfg, train_df, val_df, test_df, model_path_root / "models" / "pooled" / method,
                    fold, int(cfg.get("seed", 42)) + fold)
                test_out = test_df[["region_id", "method", "fold", "track_group_id", "scene_id", "img_row", "img_col", "Latitude", "Longitude", "Depth"]].copy()
                test_out["Prediction"] = pred
                pred_path.parent.mkdir(parents=True, exist_ok=True); test_out.to_parquet(pred_path, index=False)
                blend_info = {"type": "validation_fitted_convex_blend", "method": method, "fold": fold,
                              "objective": cfg["convex_blend"].get("objective", "rmse"),
                              "fit_split": "outer_fold_validation", "train_rows": len(train_df),
                              "validation_rows": len(val_df), **model_info}
                write_json(blend_path, blend_info)
            mm = metrics(test_out.Depth.to_numpy(float), test_out.Prediction.to_numpy(float))
            weight_cols = {f"weight_{name}": float(weight) for name, weight in zip(blend_info["base_regressors"], blend_info["weights"])}
            results.append({"scope": "pooled", "method": method, "fold": fold, "regressor": "ConvexBlend",
                            "status": "OK", "train_rows": len(train_df), "validation_rows": len(val_df),
                            "test_rows": len(test_out), **weight_cols, **mm})
            for region_id, region_group in test_out.groupby("region_id"):
                region_metrics = metrics(region_group.Depth.to_numpy(float), region_group.Prediction.to_numpy(float))
                results.append({"scope": "region", "region_id": region_id, "method": method, "fold": fold,
                                "regressor": "ConvexBlend", "status": "OK", **region_metrics})
            task_count += 1
            pd.DataFrame(results).to_csv(path / "train_metrics.csv", index=False)
            action = "reused" if reused else "trained"
            print(f"pooled convex {task_count}: {action} {method}/fold{fold} train={len(train_df)} val={len(val_df)} test={len(test_out)}", flush=True)
            if max_tasks is not None and task_count >= max_tasks:
                write_json(path / "train_summary.json", {"training_mode": "pooled", "primary_regressor": "ConvexBlend", "tasks": task_count, "rows": len(results), "status": "partial"})
                return {"tasks": task_count, "metrics": results}
    df = pd.DataFrame(results); df.to_csv(path / "train_metrics.csv", index=False)
    write_json(path / "train_summary.json", {"training_mode": "pooled", "primary_regressor": "ConvexBlend", "tasks": task_count, "rows": len(df), "status": "completed"})
    return {"tasks": task_count, "metrics": df.to_dict("records")}


def train_final_convex_blend(cfg: dict[str, Any], run_id: str, method: str = "Proposed") -> dict[str, Any]:
    """Fit one deployment ConvexBlend on all finite samples from all regions.

    Blend weights and fixed boosting/CNN training lengths come only from the
    completed outer-fold validation fits. The final base learners then see all
    available rows; no outer test fold is retained for this deployment model.
    """
    import gc
    from catboost import CatBoostRegressor
    from joblib import dump
    from experiment_2_model_selection.workflow import _models, extract_sentinel2_patches

    path = run_dir(cfg, run_id); features = list(cfg["features"])
    root = model_run_dir(cfg, run_id) / "models" / "final_full" / method
    blend_dir = root / "ConvexBlend"; blend_path = blend_dir / "final.json"
    parts = []
    for region in regions_from_config(cfg):
        source = path / "splits" / region.id / f"{method}.parquet"
        if source.exists():
            parts.append(pd.read_parquet(source))
    if not parts:
        raise FileNotFoundError(f"no split data found for final model method={method}")
    data = finite_rows(pd.concat(parts, ignore_index=True), features + ["Depth"])
    fold_infos = []
    fold_root = model_run_dir(cfg, run_id) / "models" / "pooled" / method / "ConvexBlend"
    for fold in range(int(cfg["n_folds"])):
        source = fold_root / f"fold_{fold}.json"
        if not source.exists():
            raise FileNotFoundError(f"missing fold blend metadata: {source}")
        fold_infos.append(json.loads(source.read_text(encoding="utf-8")))
    base_names = [str(value) for value in fold_infos[0]["base_regressors"]]
    if any(info["base_regressors"] != base_names for info in fold_infos):
        raise ValueError("outer folds use inconsistent ConvexBlend base regressors")
    weights = np.mean(np.asarray([info["weights"] for info in fold_infos], dtype="float64"), axis=0)
    weights = np.clip(weights, 0.0, None); weights /= weights.sum()
    catboost_iterations = max(1, int(np.median([
        int(info.get("catboost_best_iteration", 0)) + 1 for info in fold_infos
    ])))
    cnn_epochs = max(1, int(np.median([int(info.get("cnn_best_epoch", 1)) for info in fold_infos])))
    models = _models(cfg, int(cfg.get("seed", 42)))
    patches = None
    for name in base_names:
        suffix = ".cbm" if name == "CatBoost" else ".joblib"
        destination = root / name / f"final{suffix}"
        if destination.exists():
            print(f"final model reused: {name}", flush=True)
            continue
        model = models[name]; destination.parent.mkdir(parents=True, exist_ok=True)
        if name == "LightweightCNN":
            patches = extract_sentinel2_patches(data, cfg, int(cfg["lightweight_cnn"].get("image_patch_size", 11)))
            model.fit_full(patches, data.Depth.to_numpy(float), cnn_epochs)
        elif name == "CatBoost":
            params = dict(cfg["catboost"]); params["iterations"] = catboost_iterations
            params.pop("early_stopping_rounds", None)
            model = CatBoostRegressor(**params, verbose=False, eval_metric="RMSE")
            model.fit(data[features], data.Depth)
        elif name == "MLP":
            model.set_params(mlpregressor__early_stopping=False)
            model.fit(data[features], data.Depth)
        else:
            model.fit(data[features], data.Depth)
        if name == "CatBoost": model.save_model(str(destination))
        else: dump(model, destination, compress=3)
        print(f"final model completed: {name} rows={len(data)}", flush=True)
        del model; gc.collect()
        if name == "LightweightCNN": del patches; patches = None; gc.collect()
    region_rows = {str(key): int(value) for key, value in data.groupby("region_id").size().items()}
    info = {
        "type": "full_sample_final_convex_blend", "method": method,
        "fit_split": "all_finite_ten_region_samples", "fit_rows": int(len(data)),
        "region_rows": region_rows, "base_regressors": base_names, "weights": weights.tolist(),
        "weight_source": "mean_of_outer_fold_validation_weights",
        "source_folds": int(cfg["n_folds"]), "catboost_iterations": catboost_iterations,
        "cnn_epochs": cnn_epochs, "prediction_uncertainty": "not_computed",
    }
    write_json(blend_path, info)
    write_json(path / "final_model_summary.json", info)
    return info


def predict(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    import rasterio
    from rasterio.transform import rowcol
    from catboost import CatBoostRegressor
    path = run_dir(cfg, run_id); model_path_root = model_run_dir(cfg, run_id); features = list(cfg["features"]); outputs = []
    if str(cfg.get("training_mode", "regional")).lower() == "pooled":
        return predict_pooled(cfg, run_id)
    for r in regions_from_config(cfg):
        scenes, _ = scene_inventory(r)
        for m in METHODS:
            model_files = sorted((path / "models" / r.id / m).glob("fold_*.cbm"))
            if not model_files: continue
            # Use the configured representative scene for final maps.
            s = next((x for x in scenes if x["scene_id"] == r.representative_scene), None)
            if s is None: continue
            sd = Path(s["scene_dir"]); model = CatBoostRegressor(); model.load_model(model_files[0])
            with rasterio.open(sd / FEATURE_FILES[features[0]]) as src:
                profile = src.profile.copy(); height, width = src.height, src.width; crs, transform = src.crs, src.transform
            arrays = []
            for feat in features:
                with rasterio.open(sd / FEATURE_FILES[feat]) as src: arrays.append(src.read(1).astype(float))
            X = np.stack(arrays, axis=-1); valid = np.isfinite(X).all(axis=-1) & (X[:, :, :4] > 0).all(axis=-1)
            with rasterio.open(sd / "cloud_mask.tif") as src: valid &= src.read(1) == 0
            fold_preds = []
            for model_path in model_files:
                fold_model = CatBoostRegressor(); fold_model.load_model(model_path); fold_preds.append(fold_model.predict(X[valid]))
            pred = np.full((height, width), np.nan, dtype="float32"); pred[valid] = np.mean(fold_preds, axis=0).astype("float32")
            uncertainty = np.full((height, width), np.nan, dtype="float32"); uncertainty[valid] = np.std(fold_preds, axis=0).astype("float32")
            full_path = path / "models" / r.id / m / "full.cbm"
            if full_path.exists():
                full_model = CatBoostRegressor(); full_model.load_model(full_path); pred[valid] = full_model.predict(X[valid]).astype("float32")
            pred = apply_prediction_domain(pred, cfg)
            out = path / "maps" / "candidates" / r.id / f"{m}.tif"; out.parent.mkdir(parents=True, exist_ok=True)
            profile.update(dtype="float32", count=1, nodata=-9999.0, compress="DEFLATE", tiled=True)
            with rasterio.open(out, "w", **profile) as dst:
                dst.write(np.where(np.isfinite(pred), pred, -9999).astype("float32"), 1)
                dst.update_tags(region_id=r.id, method=m, scene_id=s["scene_id"], run_id=run_id)
            unc_out = path / "maps" / "uncertainty" / r.id / f"{m}.tif"; unc_out.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(unc_out, "w", **profile) as dst:
                dst.write(np.where(np.isfinite(uncertainty), uncertainty, -9999).astype("float32"), 1)
                dst.update_tags(region_id=r.id, method=m, scene_id=s["scene_id"], run_id=run_id, statistic="five_fold_std")
            qa_out = path / "maps" / "qa" / r.id / f"{m}.tif"; qa_out.parent.mkdir(parents=True, exist_ok=True)
            qa_profile = profile.copy(); qa_profile.update(dtype="uint8", nodata=0)
            with rasterio.open(qa_out, "w", **qa_profile) as dst: dst.write(np.isfinite(pred).astype("uint8"), 1)
            outputs.append({"region_id": r.id, "method": m, "path": str(out), "uncertainty": str(unc_out), "qa": str(qa_out), "valid_pixels": int(np.isfinite(pred).sum())})
    # Select one final map per region only after strict metrics are available.
    metric_file = path / "report_common_support_metrics.csv"
    if metric_file.exists():
        ranked = pd.read_csv(metric_file).sort_values(["region_id", "rmse", "mae", "bias_abs"])
        for region_id, g in ranked.groupby("region_id"):
            method = str(g.iloc[0]["method"]); item = next((x for x in outputs if x["region_id"] == region_id and x["method"] == method), None)
            if not item: continue
            for kind, key in (("depth", "path"), ("uncertainty", "uncertainty"), ("qa", "qa")):
                src = Path(item[key]); dst = path / "maps" / "final" / f"{region_id}_{kind}.tif"; dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
    write_json(path / "maps" / "summary.json", outputs); return {"maps": outputs}
    write_json(path / "maps" / "summary.json", outputs); return {"maps": outputs}


def _predict_cnn_raster_points(model: Any, raw_stack: np.ndarray, cloud: np.ndarray,
                               valid: np.ndarray, patch_size: int,
                               chunk_size: int = 20000) -> np.ndarray:
    """Predict valid raster centers from raw image windows without materializing all patches."""
    from numpy.lib.stride_tricks import sliding_window_view

    radius = int(patch_size) // 2
    padded = np.pad(raw_stack, ((0, 0), (radius, radius), (radius, radius)), mode="reflect")
    windows = sliding_window_view(padded, (int(patch_size), int(patch_size)), axis=(1, 2))
    cloud_padded = np.pad(cloud, ((radius, radius), (radius, radius)), mode="edge")
    cloud_windows = sliding_window_view(cloud_padded, (int(patch_size), int(patch_size)))
    rows, cols = np.where(valid); output = np.empty(len(rows), dtype="float64")
    for start in range(0, len(rows), int(chunk_size)):
        stop = min(start + int(chunk_size), len(rows)); rr, cc = rows[start:stop], cols[start:stop]
        patches = windows[:, rr, cc].transpose(1, 0, 2, 3).copy()
        masks = cloud_windows[rr, cc] != 0; patches = np.where(masks[:, None], np.nan, patches)
        medians = np.nanmedian(patches, axis=(2, 3), keepdims=True)
        medians[~np.isfinite(medians)] = 0.0
        patches = np.where(np.isfinite(patches), patches, medians).astype("float32")
        output[start:stop] = model.predict(patches)
    return output


def predict_pooled(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Apply shared pooled ConvexBlend fold models to representative scenes."""
    import rasterio
    from catboost import CatBoostRegressor
    from joblib import load
    from experiment_2_model_selection.workflow import _read_sentinel2_stack

    path = run_dir(cfg, run_id); model_path_root = model_run_dir(cfg, run_id); features = list(cfg["features"]); outputs = []
    for region in regions_from_config(cfg):
        scenes, _ = scene_inventory(region)
        scene = next((x for x in scenes if x["scene_id"] == region.representative_scene), None)
        if scene is None:
            continue
        scene_dir = Path(scene["scene_dir"])
        with rasterio.open(scene_dir / FEATURE_FILES[features[0]]) as src:
            profile = src.profile.copy(); height, width = src.height, src.width
        arrays = []
        for feature in features:
            with rasterio.open(scene_dir / FEATURE_FILES[feature]) as src:
                arrays.append(src.read(1).astype(float))
        feature_cube = np.stack(arrays, axis=-1)
        with rasterio.open(scene_dir / "cloud_mask.tif") as src:
            cloud = src.read(1)
        valid = np.isfinite(feature_cube).all(axis=-1) & (feature_cube[:, :, :4] > 0).all(axis=-1) & (cloud == 0)
        raw_stack = _read_sentinel2_stack(scene_dir)
        for method in METHODS:
            method_root = model_path_root / "models" / "pooled" / method
            blend_files = sorted((method_root / "ConvexBlend").glob("fold_*.json"))
            if not blend_files:
                continue
            fold_predictions = []
            for blend_file in blend_files:
                blend = json.loads(blend_file.read_text(encoding="utf-8")); fold = int(blend["fold"])
                base_predictions = []
                for name in blend["base_regressors"]:
                    if name == "CatBoost":
                        model = CatBoostRegressor(); model.load_model(str(method_root / name / f"fold_{fold}.cbm"))
                    else:
                        model = load(method_root / name / f"fold_{fold}.joblib")
                    if name == "LightweightCNN":
                        base_pred = _predict_cnn_raster_points(
                            model, raw_stack, cloud, valid,
                            int(cfg["lightweight_cnn"].get("image_patch_size", 11)))
                    else:
                        base_pred = np.asarray(model.predict(feature_cube[valid]), dtype="float64")
                    base_predictions.append(base_pred)
                fold_predictions.append(np.column_stack(base_predictions) @ np.asarray(blend["weights"], dtype="float64"))
            pred = np.full((height, width), np.nan, dtype="float32")
            uncertainty = np.full((height, width), np.nan, dtype="float32")
            pred[valid] = np.mean(fold_predictions, axis=0).astype("float32")
            uncertainty[valid] = np.std(fold_predictions, axis=0).astype("float32")
            pred = apply_prediction_domain(pred, cfg)
            profile.update(dtype="float32", count=1, nodata=-9999.0, compress="DEFLATE", tiled=True)
            out = path / "maps" / "candidates" / region.id / f"{method}.tif"; out.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out, "w", **profile) as dst:
                dst.write(np.where(np.isfinite(pred), pred, -9999).astype("float32"), 1)
                dst.update_tags(region_id=region.id, method=method, training_mode="pooled", regressor="ConvexBlend", run_id=run_id)
            unc_out = path / "maps" / "uncertainty" / region.id / f"{method}.tif"; unc_out.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(unc_out, "w", **profile) as dst:
                dst.write(np.where(np.isfinite(uncertainty), uncertainty, -9999).astype("float32"), 1)
            qa_out = path / "maps" / "qa" / region.id / f"{method}.tif"; qa_out.parent.mkdir(parents=True, exist_ok=True)
            qa_profile = profile.copy(); qa_profile.update(dtype="uint8", nodata=0)
            with rasterio.open(qa_out, "w", **qa_profile) as dst:
                dst.write(np.isfinite(pred).astype("uint8"), 1)
            outputs.append({"region_id": region.id, "method": method, "training_mode": "pooled",
                            "regressor": "ConvexBlend", "path": str(out), "uncertainty": str(unc_out),
                            "qa": str(qa_out), "valid_pixels": int(np.isfinite(pred).sum())})
    write_json(path / "maps" / "summary.json", outputs); return {"maps": outputs}


def report(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    if str(cfg.get("training_mode", "regional")).lower() == "pooled":
        return report_pooled(cfg, run_id)
    path = run_dir(cfg, run_id); metrics_path = path / "train_metrics.csv"
    if not metrics_path.exists(): raise FileNotFoundError(metrics_path)
    df = pd.read_csv(metrics_path); ok = df[df.status.eq("OK")].copy(); summary = {}
    common_rows = []
    pred_files = list((path / "predictions").glob("**/fold_*.parquet"))
    for region_id in sorted({p.parts[-3] for p in pred_files}):
        by_method = {}
        for m in METHODS:
            fs = sorted((path / "predictions" / region_id / m).glob("fold_*.parquet"))
            if fs: by_method[m] = pd.concat([pd.read_parquet(x) for x in fs], ignore_index=True)
        if len(by_method) != len(METHODS): continue
        keys = ["region_id", "fold", "track_group_id", "scene_id", "img_row", "img_col"]
        common = None
        for m, x in by_method.items():
            z = x[keys + ["Depth", "Prediction"]].rename(columns={"Depth": f"Depth_{m}", "Prediction": f"Prediction_{m}"})
            common = z if common is None else common.merge(z, on=keys, how="inner")
        if common is None or common.empty: continue
        for m in METHODS:
            mm = metrics(common[f"Depth_{m}"].to_numpy(), common[f"Prediction_{m}"].to_numpy())
            mm.update({"region_id": region_id, "method": m, "bias_abs": abs(mm["bias"]), "support_n": len(common)})
            common_rows.append(mm)
    if common_rows:
        common_df = pd.DataFrame(common_rows); common_df.to_csv(path / "report_common_support_metrics.csv", index=False)
        summary["common_support"] = common_df.to_dict("records")
    if not ok.empty:
        summary["method_region_mean"] = ok.groupby(["region_id", "method"], as_index=False)[["rmse", "mae", "bias", "r2"]].mean().to_dict("records")
        pivot = ok.pivot_table(index=["region_id", "fold"], columns="method", values="rmse", aggfunc="mean").dropna()
        if len(pivot) >= 3:
            summary["friedman"] = {"statistic": float(friedmanchisquare(*[pivot[m].to_numpy() for m in METHODS]).statistic), "pvalue": float(friedmanchisquare(*[pivot[m].to_numpy() for m in METHODS]).pvalue)}
            comparisons = []
            for m in BASE_METHODS:
                try:
                    w = wilcoxon(pivot["Proposed"], pivot[m], alternative="two-sided", method="auto")
                    comparisons.append({"method": m, "statistic": float(w.statistic), "pvalue": float(w.pvalue)})
                except ValueError: comparisons.append({"method": m, "pvalue": float("nan")})
            summary["wilcoxon"] = comparisons
    summary["verification_note"] = "Metrics are descriptive; strict common-support and CUDEM tables are emitted when all stages are available."
    write_json(path / "report_summary.json", summary)
    if ok is not None: ok.to_csv(path / "report_fold_metrics.csv", index=False)
    return summary


def _predict_saved_convex_blend(cfg: dict[str, Any], run_id: str, method: str, fold: int,
                                frame: pd.DataFrame, cnn_patches: np.ndarray) -> np.ndarray:
    """Apply one saved pooled ConvexBlend fold to a caller-supplied support.

    The caller supplies one fixed frame and one fixed CNN patch tensor, so all
    classification methods are evaluated at exactly the same Sentinel-2
    pixels.  No target values enter prediction.
    """
    from catboost import CatBoostRegressor
    from joblib import load

    features = list(cfg["features"])
    root = model_run_dir(cfg, run_id) / "models" / "pooled" / method
    blend_path = root / "ConvexBlend" / f"fold_{fold}.json"
    if not blend_path.exists():
        raise FileNotFoundError(blend_path)
    blend = json.loads(blend_path.read_text(encoding="utf-8"))
    predictions = []
    for name in blend["base_regressors"]:
        if name == "CatBoost":
            model = CatBoostRegressor(); model.load_model(str(root / name / f"fold_{fold}.cbm"))
            pred = model.predict(frame[features])
        else:
            model = load(root / name / f"fold_{fold}.joblib")
            pred = model.predict(cnn_patches if name == "LightweightCNN" else frame[features])
        predictions.append(np.asarray(pred, dtype="float64"))
    matrix = np.column_stack(predictions)
    weights = np.asarray(blend["weights"], dtype="float64")
    if matrix.shape[1] != len(weights):
        raise ValueError(f"Blend width mismatch for {method}/fold{fold}: {matrix.shape[1]} != {len(weights)}")
    result = matrix @ weights
    if len(result) != len(frame) or not np.isfinite(result).all():
        raise ValueError(f"Non-finite or incomplete anchor prediction for {method}/fold{fold}")
    return result


def evaluate_cudem_anchor(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Evaluate every pooled model on identical Proposed-anchor CUDEM samples.

    The anchor rows are the Proposed method's held-out samples for an outer
    fold.  CUDEM is sampled once at those coordinates, invalid/land/deep cells
    are removed once, and the resulting frame is then passed unchanged to all
    seven saved models.  Consequently every method has exactly the same test
    sample count and independent reference depth within each fold and region.
    """
    from experiment_2_model_selection.workflow import extract_sentinel2_patches

    path = run_dir(cfg, run_id)
    features = list(cfg["features"])
    anchor_method = str(cfg.get("shared_test_anchor_method", "Proposed"))
    region_map = {r.id: r for r in regions_from_config(cfg)}
    output_dir = path / "evaluation" / "cudem_anchor"
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows, region_rows = [], []
    patch_size = int(cfg["lightweight_cnn"].get("image_patch_size", 11))
    for fold in range(int(cfg["n_folds"])):
        anchor_parts = []
        for region_id, region_obj in region_map.items():
            split_path = path / "splits" / region_id / f"{anchor_method}.parquet"
            if not split_path.exists():
                continue
            data = pd.read_parquet(split_path)
            split_col = f"split_fold_{fold}"
            if split_col not in data.columns:
                continue
            anchor = finite_rows(data[data[split_col].eq("test")], features)
            if anchor.empty:
                continue
            reference = sample_cudem_depth(
                region_obj,
                anchor["Longitude"].to_numpy(float),
                anchor["Latitude"].to_numpy(float),
            )
            anchor = anchor.copy()
            anchor["CUDEM_Depth"] = reference
            anchor = anchor[depth_domain_mask(anchor["CUDEM_Depth"], cfg)].copy()
            if not anchor.empty:
                anchor_parts.append(anchor)
        if not anchor_parts:
            raise RuntimeError(f"No valid CUDEM anchor samples for fold {fold}")
        anchor_frame = pd.concat(anchor_parts, ignore_index=True)
        anchor_frame["anchor_method"] = anchor_method
        anchor_frame["reference"] = "CUDEM"
        anchor_file = output_dir / f"fold_{fold}_samples.parquet"
        anchor_frame.to_parquet(anchor_file, index=False)
        cnn_patches = extract_sentinel2_patches(anchor_frame, cfg, patch_size)
        for method in METHODS:
            prediction = _predict_saved_convex_blend(
                cfg, run_id, method, fold, anchor_frame, cnn_patches
            )
            overall = metrics(anchor_frame["CUDEM_Depth"].to_numpy(float), prediction)
            overall.update({"method": method, "fold": fold, "reference": "CUDEM",
                            "support": "Proposed_anchor_pixels", "support_n": len(anchor_frame)})
            fold_rows.append(overall)
            for region_id, index in anchor_frame.groupby("region_id").groups.items():
                idx = np.asarray(list(index), dtype=int)
                regional = metrics(
                    anchor_frame.iloc[idx]["CUDEM_Depth"].to_numpy(float), prediction[idx]
                )
                regional.update({"region_id": region_id, "method": method, "fold": fold,
                                 "reference": "CUDEM", "support": "Proposed_anchor_pixels",
                                 "support_n": len(idx)})
                region_rows.append(regional)
        print(f"CUDEM anchor fold {fold}: n={len(anchor_frame)} methods={len(METHODS)}", flush=True)
    fold_df = pd.DataFrame(fold_rows)
    region_df = pd.DataFrame(region_rows)
    counts = fold_df.pivot(index="fold", columns="method", values="support_n")
    if counts.isna().any().any() or not counts.nunique(axis=1).eq(1).all():
        raise RuntimeError("CUDEM anchor support is not identical across methods")
    summary_df = fold_df.groupby("method", as_index=False).agg(
        folds=("fold", "count"), support_n_total=("support_n", "sum"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        bias_mean=("bias", "mean"), bias_std=("bias", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
    )
    fold_df.to_csv(path / "report_cudem_anchor_fold_metrics.csv", index=False)
    region_df.to_csv(path / "report_cudem_anchor_region_metrics.csv", index=False)
    summary_df.to_csv(path / "report_cudem_anchor_summary.csv", index=False)
    result: dict[str, Any] = {
        "reference": "CUDEM", "support": "Proposed_anchor_pixels",
        "anchor_method": anchor_method, "identical_sample_count_across_methods": True,
        "fold_support_n": {str(int(k)): int(v) for k, v in counts.iloc[:, 0].items()},
        "method_summary": summary_df.to_dict("records"),
    }
    pivot = fold_df.pivot(index="fold", columns="method", values="rmse").dropna()
    if len(pivot) >= 3 and all(m in pivot.columns for m in METHODS):
        ftest = friedmanchisquare(*[pivot[m].to_numpy() for m in METHODS])
        result["friedman_rmse"] = {"statistic": float(ftest.statistic), "pvalue": float(ftest.pvalue)}
        comparisons = []
        for method in BASE_METHODS:
            test = wilcoxon(pivot["Proposed"], pivot[method], alternative="two-sided", method="auto")
            comparisons.append({"method": method, "statistic": float(test.statistic),
                                "pvalue": float(test.pvalue)})
        result["wilcoxon_rmse"] = comparisons
    return result


def report_pooled(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Report pooled-fold and per-region metrics without mixing training scopes."""
    path = run_dir(cfg, run_id); metrics_path = path / "train_metrics.csv"
    if not metrics_path.exists(): raise FileNotFoundError(metrics_path)
    df = pd.read_csv(metrics_path); summary = {
        "training_mode": "pooled", "primary_reference": "CUDEM",
        "primary_support": "Proposed_anchor_pixels",
    }
    summary["primary_cudem_anchor"] = evaluate_cudem_anchor(cfg, run_id)
    pooled = df[(df.scope == "pooled") & df.status.eq("OK") & (df.fold != "full")].copy()
    region = df[(df.scope == "region") & df.status.eq("OK")].copy()
    if not pooled.empty:
        summary["supplementary_icesat2_method_mean"] = pooled.groupby("method", as_index=False)[["rmse", "mae", "bias", "r2"]].mean().to_dict("records")
        pivot = pooled.pivot_table(index="fold", columns="method", values="rmse", aggfunc="mean").dropna()
        if len(pivot) >= 3 and len(METHODS) > 1:
            ftest = friedmanchisquare(*[pivot[m].to_numpy() for m in METHODS if m in pivot.columns])
            summary["supplementary_icesat2_friedman"] = {"statistic": float(ftest.statistic), "pvalue": float(ftest.pvalue)}
            summary["supplementary_icesat2_wilcoxon"] = []
            for m in BASE_METHODS:
                if "Proposed" not in pivot or m not in pivot: continue
                try:
                    w = wilcoxon(pivot["Proposed"], pivot[m], alternative="two-sided", method="auto")
                    summary["supplementary_icesat2_wilcoxon"].append({"method": m, "statistic": float(w.statistic), "pvalue": float(w.pvalue)})
                except ValueError: pass
    if not region.empty:
        region.to_csv(path / "report_region_metrics.csv", index=False)
        summary["region_method_mean"] = region.groupby(["region_id", "method"], as_index=False)[["rmse", "mae", "bias", "r2"]].mean().to_dict("records")
    pred_files = list((path / "predictions" / "pooled").glob("*/fold_*.parquet")); by_method = {}
    for p in pred_files:
        by_method.setdefault(p.parent.name, []).append(pd.read_parquet(p))
    common = None; keys = ["region_id", "fold", "track_group_id", "scene_id", "img_row", "img_col"]
    for m, files in by_method.items():
        x = pd.concat(files, ignore_index=True); z = x[keys + ["Depth", "Prediction"]].rename(columns={"Depth": f"Depth_{m}", "Prediction": f"Prediction_{m}"})
        common = z if common is None else common.merge(z, on=keys, how="inner")
    if common is not None and not common.empty:
        rows = []
        for m in by_method:
            mm = metrics(common[f"Depth_{m}"].to_numpy(), common[f"Prediction_{m}"].to_numpy()); mm.update({"method": m, "bias_abs": abs(mm["bias"]), "support_n": len(common)}); rows.append(mm)
        pd.DataFrame(rows).to_csv(path / "report_common_support_metrics.csv", index=False); summary["supplementary_icesat2_common_support"] = rows
    # External CUDEM validation is kept separate from the ICESat-2 target
    # metrics above. It uses only held-out test coordinates and never enters
    # ConvexBlend base-model fitting or validation-weight estimation.
    cudem_rows = []; paired_rows = []
    region_map = {r.id: r for r in regions_from_config(cfg)}
    for method, files in by_method.items():
        x = pd.concat(files, ignore_index=True)
        if not {"Latitude", "Longitude"}.issubset(x.columns): continue
        for rid, rg in x.groupby("region_id"):
            region_obj = region_map.get(str(rid))
            if region_obj is None: continue
            ref = sample_cudem_depth(region_obj, rg["Longitude"].to_numpy(float), rg["Latitude"].to_numpy(float))
            z = rg.copy(); z["CUDEM_Depth"] = ref
            # CUDEM contains both topography and bathymetry. Keep finite,
            # non-negative bathymetry for the full-scene protocol; no 20 m
            # upper bound is applied.
            z = z[depth_domain_mask(z["CUDEM_Depth"], cfg) & np.isfinite(z["Prediction"])]
            for fold, fg in z.groupby("fold"):
                if len(fg) < 2: continue
                cmm = metrics(fg["CUDEM_Depth"].to_numpy(float), fg["Prediction"].to_numpy(float)); cmm.update({"region_id": rid, "method": method, "fold": fold, "reference": "CUDEM", "support_n": len(fg)})
                cudem_rows.append(cmm)
                imm = metrics(fg["Depth"].to_numpy(float), fg["Prediction"].to_numpy(float))
                paired_rows.append({"region_id": rid, "method": method, "fold": fold, "paired_n": len(fg),
                                    **{f"icesat2_{k}": v for k, v in imm.items()},
                                    **{f"cudem_{k}": v for k, v in cmm.items() if k in ("rmse", "mae", "median_ae", "bias", "r2", "pearson_r2")}})
    if cudem_rows:
        cudem_df = pd.DataFrame(cudem_rows); cudem_df.to_csv(path / "report_cudem_metrics.csv", index=False)
        summary["supplementary_method_specific_cudem"] = cudem_df.groupby(["region_id", "method"], as_index=False)[["rmse", "mae", "bias", "r2", "support_n"]].mean().to_dict("records")
    if paired_rows:
        paired_df = pd.DataFrame(paired_rows); paired_df.to_csv(path / "report_paired_reference_metrics.csv", index=False)
        summary["supplementary_paired_reference_validation"] = {"same_test_support_within_method": True, "rows": int(len(paired_df)), "metrics": paired_df.groupby(["region_id", "method"], as_index=False)[["icesat2_rmse", "icesat2_mae", "cudem_rmse", "cudem_mae", "paired_n"]].mean().to_dict("records")}
    summary["verification_note"] = (
        "Primary metrics use one identical set of Proposed-anchor Sentinel-2 pixels and independent CUDEM depths for every method. "
        "Method-specific ICESat-2 and CUDEM tables are supplementary diagnostics only."
    )
    write_json(path / "report_summary.json", summary); df.to_csv(path / "report_fold_metrics.csv", index=False); return summary
