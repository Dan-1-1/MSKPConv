"""Lightweight time-window sensitivity experiment for ICESat-2/Sentinel-2 matching.

This script builds temporary matched samples with row-level time differences,
trains a small CatBoost model for 30/60/90/180 day windows, and writes summary
tables for manuscript discussion. It does not modify the main Match_Result,
trained models, or full-scene prediction outputs.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

import config
from utils import compute_metrics


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
MATCH_SCRIPT = REPO_DIR / "ICEsat-2_Sentinel-2_Match" / "match_icesat2_sentinel2_scene.py"
OUTPUT_DIR = SCRIPT_DIR / "time_window_experiment_outputs"

WINDOWS = [30, 60, 90, 180]
RANDOM_SEED = 42
TEST_WINDOW_DAYS = 180
LIGHT_CATBOOST_PARAMS = dict(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=1.0,
    random_strength=0.5,
    bagging_temperature=0.2,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=RANDOM_SEED,
    verbose=False,
    early_stopping_rounds=30,
    thread_count=-1,
    allow_writing_files=False,
)


def load_match_module():
    spec = importlib.util.spec_from_file_location("s2_is2_match", MATCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import matching script: {MATCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATCH = load_match_module()


def log(message: str) -> None:
    print(message, flush=True)


def iqr(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    q75, q25 = np.percentile(values.to_numpy(dtype="float64"), [75, 25])
    return float(q75 - q25)


def build_scene_index(region: str, selected_tif_dir_name: str) -> pd.DataFrame:
    area_dir = config.region_dir(region)
    tif_dir = area_dir / "Sentinel-2" / selected_tif_dir_name
    rows = []
    for scene_dir, s2_time in MATCH.list_complete_scenes(tif_dir):
        min_lon, min_lat, max_lon, max_lat, crs, width, height = MATCH.bounds_wgs84(
            scene_dir / MATCH.FEATURE_FILES["B02"]
        )
        split = "testing" if scene_dir.name == MATCH.TEST_SCENES_BY_AREA[region] else "training"
        rows.append(
            {
                "area": region,
                "scene_id": scene_dir.name,
                "scene_dir": str(scene_dir),
                "s2_time": pd.Timestamp(s2_time),
                "s2_season": MATCH.hydrologic_season(s2_time),
                "split": split,
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
                "crs": crs,
                "width": width,
                "height": height,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("s2_time").reset_index(drop=True)


def prediction_file_times(region: str, prediction_dir_name: str) -> pd.DataFrame:
    prediction_dir = config.region_dir(region) / prediction_dir_name
    rows = []
    for csv_path in sorted(prediction_dir.glob("*.csv")):
        is2_time = MATCH.parse_is2_time(csv_path.name)
        if is2_time is not None:
            rows.append({"path": csv_path, "is2_time": pd.Timestamp(is2_time)})
    return pd.DataFrame(rows)


def load_prediction_points_fast(
    prediction_files: pd.DataFrame,
    relevant_scene_times: list[pd.Timestamp],
    max_window: int,
) -> pd.DataFrame:
    required = {MATCH.LAT_COL, MATCH.LON_COL, MATCH.LABEL_COL, MATCH.DEPTH_COL}
    rows = []
    skipped = []

    if prediction_files.empty:
        return pd.DataFrame(columns=[MATCH.LAT_COL, MATCH.LON_COL, "Depth_IS2", "is2_time", "is2_season"])

    scene_times = pd.to_datetime(pd.Series(relevant_scene_times))
    keep_paths = []
    for _, row in prediction_files.iterrows():
        is2_time = pd.Timestamp(row["is2_time"])
        if scene_times.empty:
            continue
        min_days = float(((scene_times - is2_time).abs().dt.total_seconds() / 86400.0).min())
        if min_days <= max_window:
            keep_paths.append((Path(row["path"]), is2_time))

    log(f"  relevant ICESat-2 files within {max_window} days: {len(keep_paths)}")
    for csv_path, is2_time in keep_paths:
        try:
            header = pd.read_csv(csv_path, nrows=0)
        except Exception as exc:
            skipped.append((csv_path.name, f"header read failed: {exc}"))
            continue

        missing = sorted(required - set(header.columns))
        if missing:
            skipped.append((csv_path.name, f"missing columns: {missing}"))
            continue

        usecols = [MATCH.LAT_COL, MATCH.LON_COL, MATCH.LABEL_COL, MATCH.DEPTH_COL]
        has_corrected = {MATCH.CORRECTED_LAT_COL, MATCH.CORRECTED_LON_COL}.issubset(header.columns)
        if has_corrected:
            usecols.extend([MATCH.CORRECTED_LAT_COL, MATCH.CORRECTED_LON_COL])

        try:
            df = pd.read_csv(csv_path, usecols=usecols)
        except Exception as exc:
            skipped.append((csv_path.name, f"data read failed: {exc}"))
            continue

        for col in usecols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if has_corrected:
            valid_corrected = df[[MATCH.CORRECTED_LAT_COL, MATCH.CORRECTED_LON_COL]].notna().all(axis=1)
            df.loc[valid_corrected, MATCH.LAT_COL] = df.loc[valid_corrected, MATCH.CORRECTED_LAT_COL]
            df.loc[valid_corrected, MATCH.LON_COL] = df.loc[valid_corrected, MATCH.CORRECTED_LON_COL]

        df = df[
            (df[MATCH.LABEL_COL] == MATCH.BATHY_LABEL)
            & df[[MATCH.LAT_COL, MATCH.LON_COL, MATCH.DEPTH_COL]].notna().all(axis=1)
        ]
        df = df[(df[MATCH.DEPTH_COL] > 0) & (df[MATCH.DEPTH_COL] <= MATCH.MAX_DEPTH_M)]
        if df.empty:
            continue

        df = df.rename(columns={MATCH.DEPTH_COL: "Depth_IS2"})
        df = df[[MATCH.LAT_COL, MATCH.LON_COL, "Depth_IS2"]].copy()
        df["is2_time"] = is2_time
        df["is2_season"] = MATCH.hydrologic_season(is2_time.to_pydatetime())
        rows.append(df)

    if skipped:
        log(f"  skipped prediction files: {len(skipped)}")
    if not rows:
        return pd.DataFrame(columns=[MATCH.LAT_COL, MATCH.LON_COL, "Depth_IS2", "is2_time", "is2_season"])
    return pd.concat(rows, ignore_index=True)


def add_fast_time_fields(points: pd.DataFrame, scene_row: pd.Series) -> pd.DataFrame:
    df = points.copy()
    s2_time = pd.Timestamp(scene_row["s2_time"])
    df["area"] = scene_row["area"]
    df["scene_id"] = scene_row["scene_id"]
    df["s2_time"] = s2_time
    df["s2_season"] = scene_row["s2_season"]
    df["split"] = scene_row["split"]
    df["time_diff_days"] = (s2_time - pd.to_datetime(df["is2_time"])).dt.total_seconds() / 86400.0
    df["same_season"] = df["s2_season"].eq(df["is2_season"])
    df["Temporal_Quality"] = [
        MATCH.temporal_quality(days, same)
        for days, same in zip(df["time_diff_days"], df["same_season"])
    ]
    # Fast sensitivity mode: keep the ICESat-2 depth as the target. The main
    # production matching pipeline performs tide correction; this experiment is
    # scoped to isolate the time-window effect without rewriting Match_Result.
    df["Depth_Tide_Corrected"] = df["Depth_IS2"]
    return df[(df["Depth_Tide_Corrected"] > 0) & (df["Depth_Tide_Corrected"] <= MATCH.MAX_DEPTH_M)].copy()


def select_training_scenes(
    scene_index: pd.DataFrame,
    prediction_files: pd.DataFrame,
    max_scenes: int,
    max_window: int,
) -> pd.DataFrame:
    train_scenes = scene_index[scene_index["split"].eq("training")].copy()
    rows = []
    if prediction_files.empty:
        return pd.DataFrame()
    is2_times = pd.to_datetime(prediction_files["is2_time"])
    for _, scene in train_scenes.iterrows():
        diffs = (is2_times - pd.Timestamp(scene["s2_time"])).abs().dt.total_seconds() / 86400.0
        min_abs = float(diffs.min())
        candidate_files = int((diffs <= max_window).sum())
        if candidate_files <= 0:
            continue
        rows.append(scene.to_dict() | {"candidate_180": candidate_files, "min_abs_days": min_abs})

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    ranked = ranked.sort_values(["min_abs_days", "s2_time"]).reset_index(drop=True)
    if max_scenes <= 0 or len(ranked) <= max_scenes:
        return ranked

    selected_parts = []
    per_season = max(1, max_scenes // max(1, ranked["s2_season"].nunique()))
    for _, group in ranked.groupby("s2_season", sort=False):
        selected_parts.append(group.head(per_season))
    selected = pd.concat(selected_parts, ignore_index=True).drop_duplicates("scene_id")
    if len(selected) < max_scenes:
        remaining = ranked[~ranked["scene_id"].isin(selected["scene_id"])]
        selected = pd.concat([selected, remaining.head(max_scenes - len(selected))], ignore_index=True)
    return selected.head(max_scenes).sort_values("s2_time").reset_index(drop=True)


def sample_scene_photons(
    points: pd.DataFrame,
    scene_row: pd.Series,
    max_window: int,
    max_photons: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    candidates = MATCH.select_points_for_scene(points, scene_row)
    if candidates.empty:
        return pd.DataFrame()

    enriched = add_fast_time_fields(candidates, scene_row)
    enriched = enriched[np.abs(enriched["time_diff_days"]) <= max_window].copy()
    if enriched.empty:
        return pd.DataFrame()

    if max_photons > 0 and len(enriched) > max_photons:
        idx = rng.choice(enriched.index.to_numpy(), max_photons, replace=False)
        enriched = enriched.loc[idx].copy()

    return MATCH.read_scene_features(Path(scene_row["scene_dir"]), enriched)


def aggregate_for_window(photon_df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    if photon_df.empty:
        return pd.DataFrame()
    subset = photon_df[np.abs(photon_df["time_diff_days"]) <= window_days].copy()
    if subset.empty:
        return pd.DataFrame()
    return MATCH.aggregate_pixels(subset)


def clean_model_df(df: pd.DataFrame) -> pd.DataFrame:
    needed = config.FEATURE_COLS + [config.TARGET_COL]
    out = df.dropna(subset=needed).copy()
    finite = np.isfinite(out[needed].astype("float64")).all(axis=1)
    return out.loc[finite].reset_index(drop=True)


def fit_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, max_train_samples: int) -> dict:
    train_df = clean_model_df(train_df)
    test_df = clean_model_df(test_df)
    if max_train_samples > 0 and len(train_df) > max_train_samples:
        train_df = train_df.sample(n=max_train_samples, random_state=RANDOM_SEED).reset_index(drop=True)

    if len(train_df) < 50 or len(test_df) < 20:
        return {
            "Train_samples": int(len(train_df)),
            "Test_samples": int(len(test_df)),
            "R2": float("nan"),
            "RMSE_m": float("nan"),
            "MAE_m": float("nan"),
            "Bias_m": float("nan"),
        }

    x = train_df[config.FEATURE_COLS].astype("float32").values
    y = train_df[config.TARGET_COL].astype("float32").values
    if len(train_df) >= 500:
        x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=0.1, random_state=RANDOM_SEED)
        eval_set = (x_va, y_va)
    else:
        x_tr, y_tr = x, y
        eval_set = None

    model = CatBoostRegressor(**LIGHT_CATBOOST_PARAMS)
    if eval_set is None:
        model.fit(x_tr, y_tr)
    else:
        model.fit(x_tr, y_tr, eval_set=eval_set, use_best_model=True)

    x_test = test_df[config.FEATURE_COLS].astype("float32").values
    y_test = test_df[config.TARGET_COL].astype("float32").values
    pred = model.predict(x_test)
    metrics = compute_metrics(y_test, pred)
    return {
        "Train_samples": int(len(train_df)),
        "Test_samples": int(len(test_df)),
        "R2": metrics["R2"],
        "RMSE_m": metrics["RMSE"],
        "MAE_m": metrics["MAE"],
        "Bias_m": metrics["Bias"],
    }


def window_stats(train_photons: pd.DataFrame, train_agg: pd.DataFrame, window_days: int) -> dict:
    photons = train_photons[np.abs(train_photons["time_diff_days"]) <= window_days].copy()
    if photons.empty:
        same_ratio = float("nan")
        med_abs = float("nan")
    else:
        same_ratio = float(photons["same_season"].mean())
        med_abs = float(np.median(np.abs(photons["time_diff_days"].to_numpy(dtype="float64"))))
    return {
        "Same_season_ratio": same_ratio,
        "Median_abs_time_diff_days": med_abs,
        "Optical_IQR_NDWI": iqr(train_agg["NDWI"]) if not train_agg.empty else float("nan"),
        "Optical_IQR_Blue_Green_LogRatio": iqr(train_agg["Blue_Green_LogRatio"]) if not train_agg.empty else float("nan"),
    }


def run_region(args, region: str, rng: np.random.Generator) -> list[dict]:
    log(f"\n[Region] {region}")
    scene_index = build_scene_index(region, args.selected_tif_dir_name)
    if scene_index.empty:
        log("  no complete Sentinel-2 scenes")
        return []

    pred_files = prediction_file_times(region, args.prediction_dir_name)
    log(f"  ICESat-2 files with parsed times: {len(pred_files)}")

    selected_train = select_training_scenes(scene_index, pred_files, args.max_train_scenes, max(WINDOWS))
    if selected_train.empty:
        log("  no training scenes with samples within 180 days")
        return []
    log(f"  selected training scenes: {len(selected_train)}")

    test_scene = scene_index[
        scene_index["scene_id"].eq(MATCH.TEST_SCENES_BY_AREA[region])
    ]
    if test_scene.empty:
        log("  configured test scene not found")
        return []
    test_scene = test_scene.iloc[0]

    relevant_times = [pd.Timestamp(t) for t in selected_train["s2_time"].tolist()]
    relevant_times.append(pd.Timestamp(test_scene["s2_time"]))
    points = load_prediction_points_fast(pred_files, relevant_times, max(WINDOWS))
    if points.empty:
        log("  no valid ICESat-2 prediction points in relevant files")
        return []
    log(f"  valid relevant ICESat-2 photon rows: {len(points):,}")

    train_photon_parts = []
    for i, (_, scene) in enumerate(selected_train.iterrows(), start=1):
        log(f"    train scene {i}/{len(selected_train)}: {scene['scene_id']}")
        sampled = sample_scene_photons(points, scene, max(WINDOWS), args.max_photons_per_scene, rng)
        if not sampled.empty:
            train_photon_parts.append(sampled)
    if not train_photon_parts:
        log("  no sampled training photons")
        return []
    train_photons = pd.concat(train_photon_parts, ignore_index=True)

    log(f"  test scene: {test_scene['scene_id']}")
    test_photons = sample_scene_photons(points, test_scene, TEST_WINDOW_DAYS, args.max_test_photons, rng)
    test_agg = aggregate_for_window(test_photons, TEST_WINDOW_DAYS)
    if test_agg.empty:
        log("  no aggregated test samples")
        return []

    rows = []
    for window_days in WINDOWS:
        train_agg = aggregate_for_window(train_photons, window_days)
        stats = window_stats(train_photons, train_agg, window_days)
        metrics = fit_and_evaluate(train_agg, test_agg, args.max_train_samples)
        row = {
            "Region": region,
            "Window_days": window_days,
            "Train_scenes": int(
                train_photons[np.abs(train_photons["time_diff_days"]) <= window_days]["scene_id"].nunique()
            ),
            **metrics,
            **stats,
        }
        log(
            "    {window:>3}d | scenes={scenes:>2} train={train:>7,} test={test:>6,} "
            "RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}".format(
                window=window_days,
                scenes=row["Train_scenes"],
                train=row["Train_samples"],
                test=row["Test_samples"],
                rmse=row["RMSE_m"] if np.isfinite(row["RMSE_m"]) else float("nan"),
                mae=row["MAE_m"] if np.isfinite(row["MAE_m"]) else float("nan"),
                r2=row["R2"] if np.isfinite(row["R2"]) else float("nan"),
            )
        )
        rows.append(row)
    return rows


def add_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Delta_RMSE_vs_180d_m"] = np.nan
    df["Delta_MAE_vs_180d_m"] = np.nan
    for region, group in df.groupby("Region"):
        ref = group[group["Window_days"].eq(180)]
        if ref.empty:
            continue
        ref_rmse = float(ref.iloc[0]["RMSE_m"])
        ref_mae = float(ref.iloc[0]["MAE_m"])
        idx = df["Region"].eq(region)
        df.loc[idx, "Delta_RMSE_vs_180d_m"] = df.loc[idx, "RMSE_m"] - ref_rmse
        df.loc[idx, "Delta_MAE_vs_180d_m"] = df.loc[idx, "MAE_m"] - ref_mae
    return df


def write_outputs(df: pd.DataFrame, args) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ordered_cols = [
        "Region",
        "Window_days",
        "Train_scenes",
        "Train_samples",
        "Test_samples",
        "Same_season_ratio",
        "Median_abs_time_diff_days",
        "Optical_IQR_NDWI",
        "Optical_IQR_Blue_Green_LogRatio",
        "R2",
        "RMSE_m",
        "MAE_m",
        "Bias_m",
        "Delta_RMSE_vs_180d_m",
        "Delta_MAE_vs_180d_m",
    ]
    df = df[ordered_cols]
    csv_path = OUTPUT_DIR / "time_window_accuracy_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")

    md_path = OUTPUT_DIR / "time_window_accuracy_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Time-window sensitivity summary\n\n")
        f.write(
            "Fast sensitivity mode uses row-level temporal filtering and Sentinel-2 "
            "optical features, skips full-scene prediction, and does not overwrite "
            "main experiment outputs.\n\n"
        )
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    meta_path = OUTPUT_DIR / "time_window_experiment_metadata.json"
    metadata = {
        "windows_days": WINDOWS,
        "test_window_days": TEST_WINDOW_DAYS,
        "regions": args.regions,
        "selected_tif_dir_name": args.selected_tif_dir_name,
        "prediction_dir_name": args.prediction_dir_name,
        "max_train_scenes": args.max_train_scenes,
        "max_photons_per_scene": args.max_photons_per_scene,
        "max_test_photons": args.max_test_photons,
        "max_train_samples": args.max_train_samples,
        "random_seed": RANDOM_SEED,
        "catboost_params": LIGHT_CATBOOST_PARAMS,
        "depth_mode": "fast_no_tide_correction_Depth_IS2",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log(f"\nSummary CSV: {csv_path}")
    log(f"Markdown table: {md_path}")
    log(f"Metadata: {meta_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="+", default=config.ALL_REGIONS, choices=config.ALL_REGIONS)
    parser.add_argument("--selected-tif-dir-name", default="TIF_Final")
    parser.add_argument("--prediction-dir-name", default="Predictions_selected")
    parser.add_argument("--max-train-scenes", type=int, default=16)
    parser.add_argument("--max-photons-per-scene", type=int, default=200_000)
    parser.add_argument("--max-test-photons", type=int, default=250_000)
    parser.add_argument("--max-train-samples", type=int, default=80_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(RANDOM_SEED)
    all_rows = []
    for region in args.regions:
        all_rows.extend(run_region(args, region, rng))

    if not all_rows:
        log("No results produced.")
        return 1
    df = add_delta_columns(pd.DataFrame(all_rows))
    write_outputs(df, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
