import argparse
import math
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from tqdm import tqdm


BASE_DIR = Path(r"H:\ATL24\PhisicKPConvNet\validation_area\data")
AREAS = ["Florida Bay", "Key Largo", "Key West", "Marathon"]

PREDICTIONS_DIR_NAME = "Predictions_selected"
S2_SUBDIR = Path("Sentinel-2")
DEFAULT_SELECTED_TIF_DIR = "TIF_Final"
MATCH_ROOT = Path("Sentinel-2") / "Match_Result"

TEST_SCENES_BY_AREA = {
    "Florida Bay": "S2B_MSIL2A_20250228T160509_T17RNH_Cropped_CloudMasked",
    "Key Largo": "S2B_MSIL2A_20250320T160509_T17RNH_Cropped_CloudMasked",
    "Key West": "S2B_MSIL2A_20260107T160549_T17RMH_Cropped_CloudMasked",
    "Marathon": "S2B_MSIL2A_20260213T160509_Mosaic_Cropped_CloudMasked",
}

MAX_DEPTH_M = 20.0
BATHY_LABEL = 3
MAX_WORKERS = min(4, max(1, (os.cpu_count() or 4) - 1))
ALLOW_MISSING_TIDE = False

LABEL_COL = "pred_label"
DEPTH_COL = "Depth"
LAT_COL = "Latitude"
LON_COL = "Longitude"
CORRECTED_LAT_COL = "Corrected_Latitude"
CORRECTED_LON_COL = "Corrected_Longitude"

FEATURE_FILES = {
    "B02": "B02.tif",
    "B03": "B03.tif",
    "B04": "B04.tif",
    "B08": "B08.tif",
    "NDWI": "NDWI.tif",
    "Blue_Green_LogRatio": "Blue_Green_LogRatio.tif",
    "Blue_Red_LogRatio": "Blue_Red_LogRatio.tif",
    "Green_Red_LogRatio": "Green_Red_LogRatio.tif",
}

OUTPUT_COLUMNS = [
    "area",
    "scene_id",
    "s2_time",
    "s2_season",
    "split",
    "Latitude",
    "Longitude",
    "img_row",
    "img_col",
    "B02",
    "B03",
    "B04",
    "B08",
    "NDWI",
    "Blue_Green_LogRatio",
    "Blue_Red_LogRatio",
    "Green_Red_LogRatio",
    "Depth",
    "Sample_Quality",
    "Temporal_Quality",
    "Photon_Count",
    "Photon_Count_Filtered",
    "Depth_Std",
]

NOAA_API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_DATUM = "MLLW"
NOAA_UNITS = "metric"
NOAA_TIMEOUT = 20

ATL03_TIME_PATTERN = re.compile(r"ATL\d{2}_(\d{14})_")
S2_TIME_PATTERN = re.compile(r"_(\d{8}T\d{6})_")

TIDE_STATIONS = [
    {"id": "8723214", "name": "Virginia Key, Biscayne Bay, FL", "lat": 25.7314, "lon": -80.1618},
    {"id": "8723970", "name": "Vaca Key, Florida Bay, FL", "lat": 24.7117, "lon": -81.1050},
    {"id": "8724580", "name": "Key West, FL", "lat": 24.5557, "lon": -81.8079},
]


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_is2_time(csv_name):
    match = ATL03_TIME_PATTERN.search(csv_name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")


def parse_s2_time(scene_name):
    match = S2_TIME_PATTERN.search(scene_name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")


def hydrologic_season(dt):
    month = dt.month
    if month in (6, 7, 8):
        return "Season_A_JunAug"
    if month in (9, 10, 11):
        return "Season_B_SepNov"
    if month in (12, 1, 2):
        return "Season_C_DecFeb"
    return "Season_D_MarMay"


def temporal_quality(time_diff_days, same_season):
    abs_days = abs(time_diff_days)
    if same_season and abs_days <= 900:
        return "high"
    if same_season:
        return "medium"
    return "low"


def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def find_area_shp(area_dir):
    shp_files = sorted(area_dir.rglob("*.shp"))
    if not shp_files:
        return None
    polygon_files = [p for p in shp_files if "POLYGON" in p.name.upper()]
    return polygon_files[0] if polygon_files else shp_files[0]


def area_centroid(area_dir):
    shp_path = find_area_shp(area_dir)
    if shp_path is None:
        return None
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        return None
    if gdf.crs is None:
        raise ValueError(f"Missing CRS in shapefile: {shp_path}")
    geom = gdf.to_crs("EPSG:4326").geometry.dropna().union_all()
    centroid = geom.centroid
    return float(centroid.y), float(centroid.x)


def sorted_tide_stations(area_dir):
    centroid = area_centroid(area_dir)
    if centroid is None:
        return [station | {"distance_km": np.nan} for station in TIDE_STATIONS]
    lat, lon = centroid
    stations = []
    for station in TIDE_STATIONS:
        distance = haversine_km(lat, lon, station["lat"], station["lon"])
        stations.append(station | {"distance_km": distance})
    return sorted(stations, key=lambda item: item["distance_km"])


def tide_request(target_time, station):
    begin_date = (target_time - timedelta(minutes=30)).strftime("%Y%m%d %H:%M")
    end_date = (target_time + timedelta(minutes=30)).strftime("%Y%m%d %H:%M")
    params = {
        "product": "water_level",
        "application": "ICESat2_Sentinel2_Matching",
        "begin_date": begin_date,
        "end_date": end_date,
        "datum": NOAA_DATUM,
        "station": station["id"],
        "time_zone": "gmt",
        "units": NOAA_UNITS,
        "format": "json",
    }
    response = requests.get(NOAA_API_URL, params=params, timeout=NOAA_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if "data" not in data:
        return None
    df = pd.DataFrame(data["data"])
    if df.empty or "v" not in df.columns:
        return None
    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna(subset=["t", "v"])
    if df.empty:
        return None
    idx = (df["t"] - target_time).abs().idxmin()
    return {
        "tide": float(df.loc[idx, "v"]),
        "station_id": station["id"],
        "station_name": station["name"],
        "station_distance_km": station.get("distance_km", np.nan),
    }


def fetch_station_tide(target_time, station, tide_cache):
    """Fetch tide for a single station with caching."""
    cache_key = f"{target_time.strftime('%Y%m%d%H%M%S')}_{station['id']}"
    if cache_key in tide_cache:
        return tide_cache[cache_key]
    try:
        result = tide_request(target_time, station)
        if result is not None:
            tide_cache[cache_key] = result
            time.sleep(0.03)
        return result
    except Exception as exc:
        log(f"NOAA tide failed for station {station['id']} at {target_time}: {exc}")
        return None


def get_all_station_tides(target_time, stations, tide_cache):
    """Fetch available tides from all stations."""
    results = []
    for station in stations:
        tide = fetch_station_tide(target_time, station, tide_cache)
        if tide is not None:
            results.append({
                "lat": station["lat"],
                "lon": station["lon"],
                "tide": tide["tide"],
                "station_id": station["id"],
                "station_name": station["name"],
            })
    return results


def idw_tide(lat, lon, station_tides, power=2):
    """Inverse Distance Weighting interpolation of tide values."""
    if not station_tides:
        return None
    if len(station_tides) == 1:
        return station_tides[0]["tide"]

    weights = []
    tides = []
    for st in station_tides:
        d = haversine_km(lat, lon, st["lat"], st["lon"])
        if d < 0.001:  # essentially at station location
            return st["tide"]
        w = 1.0 / (d ** power)
        weights.append(w)
        tides.append(st["tide"])

    total_weight = sum(weights)
    if total_weight == 0 or not np.isfinite(total_weight):
        return None
    return np.average(tides, weights=weights)


def ensure_output_dirs(area_dir):
    train_dir = area_dir / MATCH_ROOT / "Train"
    test_dir = area_dir / MATCH_ROOT / "Test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    return {"train": train_dir, "test": test_dir}


def clear_previous_outputs(out_dir):
    for csv_path in out_dir.glob("Match_*.csv"):
        csv_path.unlink()


def list_complete_scenes(tif_dir):
    scenes = []
    if not tif_dir.exists():
        return scenes
    for scene_dir in sorted(tif_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        s2_time = parse_s2_time(scene_dir.name)
        if s2_time is None:
            continue
        required = [scene_dir / name for name in FEATURE_FILES.values()]
        required.append(scene_dir / "cloud_mask.tif")
        if not all(path.exists() for path in required):
            log(f"Skipping incomplete scene: {scene_dir}")
            continue
        scenes.append((scene_dir, s2_time))
    return scenes


def bounds_wgs84(raster_path):
    with rasterio.open(raster_path) as src:
        src_bounds = src.bounds
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        xs = [src_bounds.left, src_bounds.right, src_bounds.right, src_bounds.left]
        ys = [src_bounds.bottom, src_bounds.bottom, src_bounds.top, src_bounds.top]
        lons, lats = transformer.transform(xs, ys)
        return min(lons), min(lats), max(lons), max(lats), str(src.crs), src.width, src.height


def build_s2_index(area_name, tif_dir, stations, tide_cache):
    rows = []
    for scene_dir, s2_time in list_complete_scenes(tif_dir):
        min_lon, min_lat, max_lon, max_lat, crs, width, height = bounds_wgs84(scene_dir / FEATURE_FILES["B02"])
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0

        station_tides = get_all_station_tides(s2_time, stations, tide_cache)
        tide_s2_value = idw_tide(center_lat, center_lon, station_tides)

        if tide_s2_value is None:
            if not ALLOW_MISSING_TIDE:
                log(f"Skipping scene due to missing tide: {scene_dir.name}")
                continue
            tide_s2_value = np.nan

        rows.append(
            {
                "area": area_name,
                "scene_id": scene_dir.name,
                "scene_dir": str(scene_dir),
                "s2_time": s2_time,
                "s2_season": hydrologic_season(s2_time),
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
                "crs": crs,
                "width": width,
                "height": height,
                "tide_s2": tide_s2_value,
                "tide_station_count": len(station_tides),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["s2_time"] = pd.to_datetime(df["s2_time"])

    df["split"] = "training"
    test_scene = TEST_SCENES_BY_AREA.get(area_name)
    if test_scene is None:
        raise ValueError(f"No fixed test scene configured for area: {area_name}")
    if test_scene not in set(df["scene_id"]):
        raise FileNotFoundError(f"Configured test scene not found in {tif_dir}: {test_scene}")
    df.loc[df["scene_id"].eq(test_scene), "split"] = "testing"

    return df.sort_values("s2_time").reset_index(drop=True)


def load_prediction_points(prediction_dir, stations, tide_cache):
    rows = []
    skipped = []
    required = {LAT_COL, LON_COL, LABEL_COL, DEPTH_COL}

    for csv_path in sorted(prediction_dir.glob("*.csv")):
        is2_time = parse_is2_time(csv_path.name)
        if is2_time is None:
            skipped.append((csv_path.name, "cannot parse ICESat-2 time"))
            continue

        try:
            header = pd.read_csv(csv_path, nrows=0)
        except Exception as exc:
            skipped.append((csv_path.name, f"read header failed: {exc}"))
            continue

        missing = sorted(required - set(header.columns))
        if missing:
            skipped.append((csv_path.name, f"missing columns: {missing}"))
            continue

        usecols = [LAT_COL, LON_COL, LABEL_COL, DEPTH_COL]
        has_corrected_coords = {CORRECTED_LAT_COL, CORRECTED_LON_COL}.issubset(set(header.columns))
        if has_corrected_coords:
            usecols.extend([CORRECTED_LAT_COL, CORRECTED_LON_COL])

        try:
            df = pd.read_csv(csv_path, usecols=usecols)
        except Exception as exc:
            skipped.append((csv_path.name, f"read data failed: {exc}"))
            continue

        for col in usecols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if has_corrected_coords:
            valid_corrected = df[[CORRECTED_LAT_COL, CORRECTED_LON_COL]].notna().all(axis=1)
            df.loc[valid_corrected, LAT_COL] = df.loc[valid_corrected, CORRECTED_LAT_COL]
            df.loc[valid_corrected, LON_COL] = df.loc[valid_corrected, CORRECTED_LON_COL]

        df = df[(df[LABEL_COL] == BATHY_LABEL) & df[[LAT_COL, LON_COL, DEPTH_COL]].notna().all(axis=1)]
        df = df[(df[DEPTH_COL] > 0) & (df[DEPTH_COL] <= MAX_DEPTH_M)]
        if df.empty:
            skipped.append((csv_path.name, "no valid bathymetric rows"))
            continue

        df = df.rename(columns={DEPTH_COL: "Depth_IS2"})
        df = df[[LAT_COL, LON_COL, "Depth_IS2"]].copy()
        df["is2_time"] = is2_time
        df["is2_season"] = hydrologic_season(is2_time)

        # Spatially-interpolated tide (IDW) for each ICESat-2 point
        station_tides = get_all_station_tides(is2_time, stations, tide_cache)
        if not station_tides:
            if not ALLOW_MISSING_TIDE:
                skipped.append((csv_path.name, "cannot fetch ICESat-2 tide from any station"))
                continue
            df["tide_is2"] = np.nan
        else:
            lats = df[LAT_COL].to_numpy()
            lons = df[LON_COL].to_numpy()
            tide_values = []
            for i in range(len(df)):
                t = idw_tide(lats[i], lons[i], station_tides)
                tide_values.append(t if t is not None else np.nan)
            df["tide_is2"] = tide_values

        rows.append(df[[LAT_COL, LON_COL, "Depth_IS2", "is2_time", "is2_season", "tide_is2"]])

    if skipped:
        log(f"  Skipped ICESat-2 files: {len(skipped)}")

    if not rows:
        return pd.DataFrame(columns=[LAT_COL, LON_COL, "Depth_IS2", "is2_time", "is2_season", "tide_is2"])

    return pd.concat(rows, ignore_index=True)


def select_points_for_scene(points, scene_row):
    pad = 0.002
    mask = (
        (points[LON_COL] >= scene_row["min_lon"] - pad)
        & (points[LON_COL] <= scene_row["max_lon"] + pad)
        & (points[LAT_COL] >= scene_row["min_lat"] - pad)
        & (points[LAT_COL] <= scene_row["max_lat"] + pad)
    )
    return points.loc[mask].copy()


def add_scene_time_fields(points, scene_row):
    df = points.copy()
    s2_time = pd.Timestamp(scene_row["s2_time"]).to_pydatetime()
    df["scene_id"] = scene_row["scene_id"]
    df["s2_time"] = s2_time
    df["s2_season"] = scene_row["s2_season"]
    df["split"] = scene_row["split"]
    df["time_diff_days"] = (pd.to_datetime(df["s2_time"]) - pd.to_datetime(df["is2_time"])).dt.total_seconds() / 86400.0
    df["same_season"] = df["s2_season"].eq(df["is2_season"])
    df["Temporal_Quality"] = [temporal_quality(days, same) for days, same in zip(df["time_diff_days"], df["same_season"])]
    df["tide_s2"] = scene_row["tide_s2"]

    has_tide = np.isfinite(df["tide_s2"]) & np.isfinite(df["tide_is2"])
    if not ALLOW_MISSING_TIDE and not bool(has_tide.all()):
        df = df.loc[has_tide].copy()
        if df.empty:
            return df
        has_tide = np.isfinite(df["tide_s2"]) & np.isfinite(df["tide_is2"])

    # Original Depth_IS2 is relative to instantaneous water surface.
    # Correct to Sentinel-2 acquisition epoch using spatially-interpolated tidal difference.
    df["tide_delta"] = np.where(has_tide, df["tide_s2"] - df["tide_is2"], 0.0)
    df["Depth_Tide_Corrected"] = df["Depth_IS2"] + df["tide_delta"]
    df = df[np.isfinite(df["Depth_Tide_Corrected"])]
    return df[(df["Depth_Tide_Corrected"] > 0) & (df["Depth_Tide_Corrected"] <= MAX_DEPTH_M)].copy()


def filter_depths_mad(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return finite
    finite_values = values[finite]
    median = np.median(finite_values)
    mad = np.median(np.abs(finite_values - median))
    if mad == 0 or not np.isfinite(mad):
        std = np.std(finite_values)
        if std == 0 or not np.isfinite(std):
            return finite
        return finite & (np.abs(values - np.mean(finite_values)) <= 3.0 * std)
    robust_sigma = 1.4826 * mad
    return finite & (np.abs(values - median) <= 3.0 * robust_sigma)


def sample_quality(filtered_count, depth_std, temporal_quality_value):
    if filtered_count >= 3 and (not np.isfinite(depth_std) or depth_std <= 1.0) and temporal_quality_value == "high":
        return "high"
    if filtered_count >= 2 and (not np.isfinite(depth_std) or depth_std <= 1.5) and temporal_quality_value in {"high", "medium"}:
        return "medium"
    return "low"


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def read_scene_features(scene_dir, points_df):
    template_path = scene_dir / FEATURE_FILES["B02"]
    with rasterio.open(template_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = transformer.transform(points_df[LON_COL].to_numpy(), points_df[LAT_COL].to_numpy())
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        in_bounds = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)

    matched = points_df.loc[in_bounds].copy()
    matched["img_row"] = rows[in_bounds]
    matched["img_col"] = cols[in_bounds]
    if matched.empty:
        return matched

    sample_rows = matched["img_row"].to_numpy(dtype=int)
    sample_cols = matched["img_col"].to_numpy(dtype=int)
    for feature, filename in FEATURE_FILES.items():
        with rasterio.open(scene_dir / filename) as src:
            arr = src.read(1)
            values = arr[sample_rows, sample_cols].astype(float)
            nodata = src.nodata
            if nodata is not None:
                values[values == nodata] = np.nan
            matched[feature] = values

    with rasterio.open(scene_dir / "cloud_mask.tif") as src:
        cloud = src.read(1)
        matched["cloud_mask"] = cloud[sample_rows, sample_cols]

    valid = matched["cloud_mask"].eq(0)
    for feature in FEATURE_FILES:
        valid &= np.isfinite(matched[feature])
    valid &= (matched["B02"] > 0) & (matched["B03"] > 0) & (matched["B04"] > 0) & (matched["B08"] > 0)
    return matched.loc[valid].copy()


def aggregate_pixels(matched_df):
    records = []
    group_cols = ["scene_id", "img_row", "img_col"]
    for _, group in matched_df.groupby(group_cols, sort=False):
        depths = group["Depth_Tide_Corrected"].to_numpy(dtype=float)
        keep = filter_depths_mad(depths)
        filtered = depths[keep]
        if filtered.size == 0:
            continue
        if filtered.size < 3:
            continue

        first = group.iloc[0]
        depth_std = float(np.std(filtered, ddof=1)) if filtered.size > 1 else np.nan
        temporal_values = group["Temporal_Quality"].tolist()
        temporal_quality_value = (
            "high" if all(v == "high" for v in temporal_values)
            else "medium" if any(v in {"high", "medium"} for v in temporal_values)
            else "low"
        )
        records.append(
            {
                "area": first["area"],
                "scene_id": first["scene_id"],
                "s2_time": first["s2_time"],
                "s2_season": first["s2_season"],
                "split": first["split"],
                "Latitude": float(group[LAT_COL].mean()),
                "Longitude": float(group[LON_COL].mean()),
                "img_row": int(first["img_row"]),
                "img_col": int(first["img_col"]),
                "B02": float(first["B02"]),
                "B03": float(first["B03"]),
                "B04": float(first["B04"]),
                "B08": float(first["B08"]),
                "NDWI": float(first["NDWI"]),
                "Blue_Green_LogRatio": float(first["Blue_Green_LogRatio"]),
                "Blue_Red_LogRatio": float(first["Blue_Red_LogRatio"]),
                "Green_Red_LogRatio": float(first["Green_Red_LogRatio"]),
                "Depth": float(np.median(filtered)),
                "Sample_Quality": sample_quality(filtered.size, depth_std, temporal_quality_value),
                "Temporal_Quality": temporal_quality_value,
                "Photon_Count": int(len(group)),
                "Photon_Count_Filtered": int(filtered.size),
                "Depth_Std": depth_std,
            }
        )
    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[OUTPUT_COLUMNS].copy()


def process_scene(scene_row, points, out_dir):
    scene_dir = Path(scene_row["scene_dir"])
    candidates = select_points_for_scene(points, scene_row)
    if candidates.empty:
        return 0

    # Add tide information for the ICESat-2 points in this scene.
    enriched = add_scene_time_fields(candidates, scene_row)
    if enriched.empty:
        return 0

    # Discard matches with time difference > 900 days
    enriched = enriched[np.abs(enriched["time_diff_days"]) <= 900].copy()
    if enriched.empty:
        return 0

    enriched["area"] = scene_row["area"]

    matched = read_scene_features(scene_dir, enriched)
    if matched.empty:
        return 0

    aggregated = aggregate_pixels(matched)
    if aggregated.empty:
        return 0

    out_stem = f"Match_{scene_row['area'].replace(' ', '_')}_{pd.Timestamp(scene_row['s2_time']).strftime('%Y%m%dT%H%M%S')}"
    aggregated.to_csv(out_dir / f"{out_stem}.csv", index=False, encoding="utf-8-sig")
    return int(len(aggregated))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match ICESat-2 bathymetric points with manually selected Sentinel-2 scenes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--areas",
        nargs="+",
        choices=AREAS,
        default=None,
        metavar="AREA",
        help=f"Areas to process. Default: all areas ({', '.join(AREAS)})",
    )
    parser.add_argument(
        "--selected-tif-dir-name",
        default=DEFAULT_SELECTED_TIF_DIR,
        help="Selected Sentinel-2 folder name under each area. Default: TIF_Final",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Match_*.csv files in Train/Test folders before processing.",
    )
    return parser.parse_args()


def process_area(area_name, selected_tif_dir_name, force=False):
    area_dir = BASE_DIR / area_name
    prediction_dir = area_dir / PREDICTIONS_DIR_NAME
    tif_dir = area_dir / S2_SUBDIR / selected_tif_dir_name
    out_dirs = ensure_output_dirs(area_dir)

    log(f"Area: {area_name}")
    if force:
        clear_previous_outputs(out_dirs["train"])
        clear_previous_outputs(out_dirs["test"])

    if not prediction_dir.exists():
        log(f"  Missing Predictions folder: {prediction_dir}")
        return 0
    if not tif_dir.exists():
        log(f"  Missing Sentinel-2 folder: {tif_dir}")
        return 0

    stations = sorted_tide_stations(area_dir)
    log(f"  Nearest tide station: {stations[0]['id']} {stations[0]['name']} ({stations[0]['distance_km']:.1f} km)")

    tide_cache = {}
    s2_index = build_s2_index(area_name, tif_dir, stations, tide_cache)
    if s2_index.empty:
        log("  No complete Sentinel-2 scenes found.")
        return 0

    points = load_prediction_points(prediction_dir, stations, tide_cache)
    if points.empty:
        log("  No valid ICESat-2 bathymetric points found.")
        return 0

    split_counts = {"training": 0, "testing": 0}
    for _, scene_row in tqdm(s2_index.iterrows(), total=len(s2_index), desc=f"{area_name} scene matching", unit="scene"):
        split = scene_row["split"]
        out_dir = out_dirs["test"] if split == "testing" else out_dirs["train"]
        count = process_scene(scene_row, points, out_dir)
        split_counts[split] += count

    log(f"  Train CSV samples: {split_counts['training']}")
    log(f"  Test  CSV samples: {split_counts['testing']}")
    log(f"  Output folders: {out_dirs['train']} ; {out_dirs['test']}")
    return split_counts["training"] + split_counts["testing"]


def main():
    args = parse_args()
    areas_to_process = args.areas if args.areas else AREAS

    log("=" * 80)
    log("ICESat-2 & Sentinel-2 Scene Matching")
    log("=" * 80)
    log(f"Base directory: {BASE_DIR}")
    log(f"Areas to process: {', '.join(areas_to_process)}")
    log(f"Max workers per area: {MAX_WORKERS}")
    log(f"Selected TIF folder: {args.selected_tif_dir_name}")
    log("Output mode: Train/Test CSV only (fixed test scenes)")
    log("=" * 80)

    total_samples = 0
    for idx, area_name in enumerate(areas_to_process, 1):
        log(f"\n[{idx}/{len(areas_to_process)}] Processing area: {area_name}")
        total_samples += process_area(area_name, args.selected_tif_dir_name, force=args.force)

    log("\n" + "=" * 80)
    log("Processing completed successfully!")
    log(f"Total matched samples written: {total_samples}")
    log("=" * 80)


if __name__ == "__main__":
    main()
