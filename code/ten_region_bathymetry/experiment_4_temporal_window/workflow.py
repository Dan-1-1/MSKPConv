"""Controlled temporal-window, sample-size, and seasonal-bias experiment.

The maximum window is matched once. Smaller windows are deterministic nested
subsets selected by the recorded absolute ICESat-2/Sentinel-2 time difference.
All variants are evaluated on the same spatially isolated main-experiment test
rows with CUDEM as the primary reference.
"""
from __future__ import annotations

import json
import math
import gc
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from experiment_1_main.workflow import (  # noqa: E402
    _fit_convex_blend_fold,
    aggregate,
    csv_files,
    depth_domain_mask,
    finite_rows,
    load_track_selection,
    load_config,
    method_columns,
    metrics,
    nearest_scene,
    parse_track,
    regions_from_config,
    sample_cudem_depth,
    sample_scene,
    selected_csv_files,
    scene_inventory,
    stable_order,
    TideProvider,
    write_json,
)


def _fit_full_sample_convex_blend(main_cfg: dict[str, Any], train: pd.DataFrame,
                                  validation: pd.DataFrame, test: pd.DataFrame,
                                  model_root: Path, seed: int,
                                  save_models: bool) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select blend weights on held-out tracks, then refit bases on all available rows."""
    from catboost import CatBoostRegressor
    from joblib import dump
    from experiment_2_model_selection.workflow import _models, extract_sentinel2_patches

    _, weights, tuning = _fit_convex_blend_fold(
        main_cfg, train, validation, test, model_root, 0, seed, False)
    full = pd.concat([train, validation], ignore_index=True)
    features = list(main_cfg["features"])
    base_names = [str(value) for value in tuning["base_regressors"]]
    models = _models(main_cfg, int(seed))
    full_predictions: list[np.ndarray] = []
    patch_size = int(main_cfg["lightweight_cnn"].get("image_patch_size", 11))
    patches = None
    if "LightweightCNN" in base_names:
        combined = pd.concat([full, test], ignore_index=True)
        patches = extract_sentinel2_patches(combined, main_cfg, patch_size)
    for name in base_names:
        model = models[name]
        if name == "LightweightCNN":
            epochs = max(1, int(tuning.get("cnn_best_epoch", 1)))
            model.fit_full(patches[:len(full)], full.Depth.to_numpy(float), epochs)
            prediction = model.predict(patches[len(full):])
        elif name == "CatBoost":
            params = dict(main_cfg["catboost"])
            params["iterations"] = max(1, int(tuning.get("catboost_best_iteration", 0)) + 1)
            params.pop("early_stopping_rounds", None)
            model = CatBoostRegressor(**params, verbose=False, eval_metric="RMSE")
            model.fit(full[features], full.Depth)
            prediction = model.predict(test[features])
        else:
            if name == "MLP":
                model.set_params(mlpregressor__early_stopping=False)
            model.fit(full[features], full.Depth)
            prediction = model.predict(test[features])
        full_predictions.append(np.asarray(prediction, dtype="float64"))
        if save_models:
            destination = model_root / name / "final"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name == "CatBoost":
                model.save_model(str(destination.with_suffix(".cbm")))
            else:
                dump(model, destination.with_suffix(".joblib"), compress=3)
        del model
        gc.collect()
    matrix = np.column_stack(full_predictions)
    tuning.update({"fit_rows": int(len(full)), "fit_split": "all_non_test_rows_after_weight_selection",
                   "refit_base_models": True})
    return matrix @ weights, weights, tuning


def load_experiment_config(path: Path | None = None) -> dict[str, Any]:
    with (path or HERE / "config.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def experiment_root(main_cfg: dict[str, Any], run_id: str, cfg: dict[str, Any]) -> Path:
    root = Path(main_cfg["output_root"]) / run_id / "experiment_4_temporal_window" / str(cfg["experiment_id"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def _main_split(main_cfg: dict[str, Any], run_id: str, region_id: str, method: str) -> Path:
    return Path(main_cfg["output_root"]) / run_id / "splits" / region_id / f"{method}.parquet"


def _master_path(root: Path, region_id: str) -> Path:
    return root / "prepared" / "master" / region_id / "Proposed.parquet"


def hemisphere_season(month: int, latitude: float) -> str:
    north = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
             6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    south = {"DJF": "JJA", "MAM": "SON", "JJA": "DJF", "SON": "MAM"}
    value = north[int(month)]
    return value if latitude >= 0 else south[value]


def add_temporal_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    track = pd.to_datetime(out["track_time"])
    scene = pd.to_datetime(out["scene_time"])
    out["time_delta_days"] = (scene - track).abs().dt.total_seconds() / 86400.0
    out["track_month"] = track.dt.month.astype(int)
    out["scene_month"] = scene.dt.month.astype(int)
    out["track_season"] = [hemisphere_season(m, lat) for m, lat in zip(out.track_month, out.Latitude)]
    out["scene_season"] = [hemisphere_season(m, lat) for m, lat in zip(out.scene_month, out.Latitude)]
    out["season_mismatch"] = out.track_season.ne(out.scene_season)
    return out


def _prepare_region(main_cfg: dict[str, Any], cfg: dict[str, Any], run_id: str,
                    region, tide: TideProvider, root: Path,
                    selection: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    destination = _master_path(root, region.id)
    if cfg.get("resume", True) and destination.exists():
        data = pd.read_parquet(destination)
        return {"region_id": region.id, "rows": len(data), "tracks": int(data.track_group_id.nunique()), "reused": True}
    # The main matching stage already produced the maximum-window Proposed
    # table with the selected-track manifest and its vertical corrections.
    # Reuse it here so the temporal experiment changes only the time-window
    # subset, rather than silently rematching all historical CSV files.
    main_prepared = (Path(main_cfg["output_root"]) / run_id / "prepared" /
                     region.id / "Proposed.parquet")
    if main_prepared.exists():
        data = pd.read_parquet(main_prepared)
        if not data.empty:
            data = add_temporal_columns(data)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(destination, index=False)
        summary = {"region_id": region.id, "rows": len(data),
                   "tracks": int(data.track_group_id.nunique()) if not data.empty else 0,
                   "complete_scenes": len(scene_inventory(region)[0]),
                   "excluded_scenes": len(scene_inventory(region)[1]),
                   "errors": [], "source": str(main_prepared), "reused": True,
                   "track_selection_manifest": main_cfg.get("matching_track_selection_csv")}
        write_json(destination.with_name("summary.json"), summary)
        return summary
    method = str(cfg["method"]); columns = method_columns(region, method)
    scenes, excluded = scene_inventory(region); records, errors = [], []
    maximum = max(int(x) for x in cfg["natural_windows_days"])
    region_files = selected_csv_files(region, selection)
    selection_by_path = {str(Path(item["path"]).resolve()): item
                         for item in (selection or {}).get(region.id, [])}
    for path in region_files:
        selection_record = selection_by_path.get(str(path.resolve()), {})
        track_id, track_dt = parse_track(path)
        if track_id is None:
            errors.append({"file": path.name, "error": "unparseable ATL03 filename"}); continue
        scene = nearest_scene(scenes, track_dt, maximum)
        if scene is None:
            errors.append({"file": path.name, "error": f"no complete scene within {maximum} days"}); continue
        scene_dt = pd.Timestamp(scene["scene_time"]).to_pydatetime()
        try:
            tide_is2, _ = tide.value(region.tide_stations, track_dt)
            tide_s2, _ = tide.value(region.tide_stations, scene_dt)
        except Exception as exc:
            errors.append({"file": path.name, "scene_id": scene["scene_id"], "error": str(exc)}); continue
        needed = {"Latitude", "Longitude", *columns.values()}
        try:
            raw = pd.read_csv(path, usecols=lambda name: name in needed, low_memory=False)
        except Exception as exc:
            errors.append({"file": path.name, "error": str(exc)}); continue
        if not all(name in raw for name in columns.values()):
            errors.append({"file": path.name, "error": "missing Proposed columns"}); continue
        selected = raw.loc[pd.to_numeric(raw[columns["label"]], errors="coerce").eq(3)].copy()
        for name in (columns["lat"], columns["lon"], columns["depth"]):
            selected[name] = pd.to_numeric(selected[name], errors="coerce")
        selected = selected.dropna(subset=[columns["lat"], columns["lon"], columns["depth"]])
        selected = selected[depth_domain_mask(selected[columns["depth"]], main_cfg, prefix="depth_match")]
        if selected.empty:
            continue
        sampled = sample_scene(Path(scene["scene_dir"]), selected[columns["lat"]].to_numpy(),
                               selected[columns["lon"]].to_numpy(), list(main_cfg["features"]))
        if sampled.empty:
            continue
        selected = selected.iloc[sampled.pop("_source_index").to_numpy(dtype=int)].reset_index(drop=True)
        depth_offset = float(selection_record.get("tutuila_depth_offset_m", 0.0) or 0.0)
        sampled["Depth"] = (selected[columns["depth"]].to_numpy(float)
                            + tide_s2 - tide_is2 + depth_offset)
        sampled["Latitude"] = selected[columns["lat"]].to_numpy(float)
        sampled["Longitude"] = selected[columns["lon"]].to_numpy(float)
        sampled["track_group_id"] = track_id; sampled["scene_id"] = scene["scene_id"]
        sampled["track_time"] = track_dt.isoformat(); sampled["scene_time"] = scene_dt.isoformat()
        sampled["region_id"] = region.id; sampled["method"] = method
        grouped = aggregate(sampled, list(main_cfg["features"]))
        if not grouped.empty:
            records.append(add_temporal_columns(grouped))
    data = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    destination.parent.mkdir(parents=True, exist_ok=True); data.to_parquet(destination, index=False)
    summary = {"region_id": region.id, "rows": len(data),
               "tracks": int(data.track_group_id.nunique()) if not data.empty else 0,
               "complete_scenes": len(scenes), "excluded_scenes": len(excluded), "errors": errors,
               "maximum_window_days": maximum, "reused": False}
    write_json(destination.with_name("summary.json"), summary)
    return summary


def qc(main_cfg: dict[str, Any], cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    root = experiment_root(main_cfg, run_id, cfg); errors, checks = [], []
    windows = [int(x) for x in cfg["natural_windows_days"]]
    if windows != sorted(set(windows)):
        errors.append("natural_windows_days must be sorted and unique")
    if int(cfg["equal_sample_reference_window_days"]) not in cfg["equal_sample_windows_days"]:
        errors.append("equal-sample reference window is absent")
    if cfg["equal_sample_windows_days"] != cfg["natural_windows_days"]:
        errors.append("natural and equal-sample windows must be identical")
    if len(cfg["repeat_seeds"]) != 3:
        errors.append("equal-sample analysis requires exactly three repeat seeds")
    regions = regions_from_config(main_cfg)
    selection = load_track_selection(main_cfg, regions)
    for region in regions:
        split = _main_split(main_cfg, run_id, region.id, str(cfg["method"]))
        scenes, excluded = scene_inventory(region)
        input_files = selected_csv_files(region, selection or None)
        item = {"region_id": region.id, "csv_count": len(input_files),
                "track_selection_manifest": main_cfg.get("matching_track_selection_csv"),
                "complete_scenes": len(scenes),
                "excluded_scenes": len(excluded), "main_split": str(split), "main_split_exists": split.exists()}
        if not split.exists(): errors.append(f"missing fixed test split: {split}")
        if not scenes: errors.append(f"no complete scenes: {region.id}")
        checks.append(item)
    result = {"status": "PASS" if not errors else "FAIL", "run_id": run_id,
              "windows_days": windows, "checks": checks, "errors": errors}
    write_json(root / "qc" / "summary.json", result)
    if errors: raise RuntimeError("experiment 4 QC failed: " + "; ".join(errors[:5]))
    return result


def prepare(main_cfg: dict[str, Any], cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    root = experiment_root(main_cfg, run_id, cfg)
    tide = TideProvider(root / "prepared" / "noaa_tide_cache.json")
    regions = regions_from_config(main_cfg)
    selection = load_track_selection(main_cfg, regions)
    summaries = [_prepare_region(main_cfg, cfg, run_id, region, tide, root, selection or None)
                 for region in regions]
    result = {"run_id": run_id, "method": cfg["method"], "regions": summaries,
              "rows": int(sum(x["rows"] for x in summaries)),
              "tracks": int(sum(x["tracks"] for x in summaries))}
    write_json(root / "prepared" / "summary.json", result)
    return result


def _load_master(main_cfg: dict[str, Any], root: Path) -> pd.DataFrame:
    parts = [_master_path(root, r.id) for r in regions_from_config(main_cfg)]
    missing = [str(path) for path in parts if not path.exists()]
    if missing: raise RuntimeError("run prepare first; missing " + ", ".join(missing[:3]))
    return pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)


def _fixed_test(main_cfg: dict[str, Any], cfg: dict[str, Any], run_id: str, fold: int) -> pd.DataFrame:
    frames = []
    region_map = {r.id: r for r in regions_from_config(main_cfg)}
    for region_id, region in region_map.items():
        frame = pd.read_parquet(_main_split(main_cfg, run_id, region_id, str(cfg["method"])))
        test = frame[frame[f"split_fold_{fold}"].eq("test")].copy()
        test["CUDEM_Depth"] = sample_cudem_depth(region, test.Longitude.to_numpy(float), test.Latitude.to_numpy(float))
        test = test[depth_domain_mask(test.CUDEM_Depth, main_cfg)].copy()
        frames.append(add_temporal_columns(test))
    return finite_rows(pd.concat(frames, ignore_index=True), list(main_cfg["features"]) + ["Depth", "CUDEM_Depth"])


def _exclude_test_buffer(data: pd.DataFrame, test: pd.DataFrame, main_cfg: dict[str, Any],
                         buffer_m: float) -> pd.DataFrame:
    kept = []
    region_map = {r.id: r for r in regions_from_config(main_cfg)}
    test_groups = set(test.track_group_id.astype(str))
    for region_id, group in data.groupby("region_id", sort=False):
        candidate = group[~group.track_group_id.astype(str).isin(test_groups)].copy()
        target = test[test.region_id.eq(region_id)]
        if candidate.empty or target.empty:
            kept.append(candidate); continue
        transformer = Transformer.from_crs("EPSG:4326", region_map[region_id].local_crs, always_xy=True)
        cx, cy = transformer.transform(candidate.Longitude.to_numpy(), candidate.Latitude.to_numpy())
        tx, ty = transformer.transform(target.Longitude.to_numpy(), target.Latitude.to_numpy())
        distance, _ = cKDTree(np.column_stack([tx, ty])).query(np.column_stack([cx, cy]), k=1)
        kept.append(candidate[np.asarray(distance) >= float(buffer_m)].copy())
    return pd.concat(kept, ignore_index=True)


def _validation_groups(data: pd.DataFrame, fraction: float, seed: int) -> set[str]:
    sizes = data.groupby(data.track_group_id.astype(str)).size().to_dict()
    target = max(1, int(round(len(data) * float(fraction))))
    ordered = sorted(sizes, key=lambda group: stable_order(group, seed))
    selected, count = [], 0
    for group in ordered:
        selected.append(group); count += int(sizes[group])
        if count >= target: break
    return set(selected)


def _strata(frame: pd.DataFrame, bins: list[float]) -> pd.Series:
    depth = pd.cut(frame.Depth, bins=[-np.inf, *bins[1:], np.inf], labels=False).astype(str)
    return frame.region_id.astype(str) + "|" + frame.track_season.astype(str) + "|d" + depth


def select_stratified_track_groups(data: pd.DataFrame, target_rows: int, seed: int,
                                   depth_bins: list[float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select complete tracks while preserving region-depth-season composition."""
    if data.empty or target_rows <= 0:
        return data.iloc[0:0].copy(), {"groups": [], "target_rows": target_rows, "actual_rows": 0}
    work = data.copy(); work["_stratum"] = _strata(work, depth_bins)
    labels = sorted(work._stratum.unique()); index = {label: i for i, label in enumerate(labels)}
    target_hist = np.zeros(len(labels), float)
    for label, count in work._stratum.value_counts().items(): target_hist[index[label]] = count
    target_hist /= target_hist.sum()
    groups = []
    for gid, group in work.groupby(work.track_group_id.astype(str), sort=False):
        hist = np.zeros(len(labels), float)
        for label, count in group._stratum.value_counts().items(): hist[index[label]] = count
        groups.append((gid, group, len(group), hist))
    rng = np.random.default_rng(seed); remaining = [groups[i] for i in rng.permutation(len(groups))]
    chosen, rows, hist = [], 0, np.zeros(len(labels), float)
    while remaining and rows < target_rows:
        scored = []
        for i, (gid, group, size, group_hist) in enumerate(remaining):
            proposed = rows + size
            composition = (hist + group_hist) / max(proposed, 1)
            score = (abs(proposed - target_rows) / max(target_rows, 1)
                     + 0.75 * np.abs(composition - target_hist).sum() / 2
                     + 0.25 * max(0, proposed - target_rows) / max(target_rows, 1))
            scored.append((score, stable_order(gid, seed), i))
        _, _, selected = min(scored)
        gid, group, size, group_hist = remaining.pop(selected)
        chosen.append(gid); rows += size; hist += group_hist
    result = work[work.track_group_id.astype(str).isin(chosen)].drop(columns="_stratum")
    return result, {"groups": chosen, "target_rows": int(target_rows), "actual_rows": len(result)}


def _metric_rows(test: pd.DataFrame, prediction: np.ndarray, base: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    overall = metrics(test.CUDEM_Depth.to_numpy(float), prediction)
    output.append({**base, "scope": "overall", "region_id": "ALL", "season": "ALL", **overall})
    evaluated = test.copy(); evaluated["Prediction"] = prediction
    for region_id, group in evaluated.groupby("region_id"):
        output.append({**base, "scope": "region", "region_id": region_id, "season": "ALL",
                       **metrics(group.CUDEM_Depth.to_numpy(float), group.Prediction.to_numpy(float))})
    for (region_id, season), group in evaluated.groupby(["region_id", "track_season"]):
        if len(group) < 2: continue
        output.append({**base, "scope": "region_season", "region_id": region_id, "season": season,
                       **metrics(group.CUDEM_Depth.to_numpy(float), group.Prediction.to_numpy(float))})
    return output


def _task_path(root: Path, design: str, window: int, fold: int, seed: int) -> Path:
    return root / "metrics" / "tasks" / f"{design}__w{window:03d}__fold{fold}__seed{seed}.json"


def run(main_cfg: dict[str, Any], cfg: dict[str, Any], run_id: str, max_tasks: int | None = None,
        windows: list[int] | None = None, folds: list[int] | None = None,
        seeds: list[int] | None = None) -> dict[str, Any]:
    root = experiment_root(main_cfg, run_id, cfg); master = _load_master(main_cfg, root)
    features = list(main_cfg["features"]); buffer_m = float(cfg["spatial_buffer_m"])
    natural_windows = [int(x) for x in cfg["natural_windows_days"] if windows is None or int(x) in windows]
    equal_windows = [int(x) for x in cfg["equal_sample_windows_days"] if windows is None or int(x) in windows]
    configured_fold = int(cfg["fixed_test_fold"])
    fold_ids = folds or [configured_fold]
    if any(int(value) != configured_fold for value in fold_ids):
        raise ValueError(f"experiment 4 uses only fixed_test_fold={configured_fold}")
    repeat_seeds = seeds or [int(x) for x in cfg["repeat_seeds"]]
    all_rows, completed = [], 0
    for fold in fold_ids:
        test = _fixed_test(main_cfg, cfg, run_id, fold)
        eligible = {window: _exclude_test_buffer(master[master.time_delta_days.le(window)], test, main_cfg, buffer_m)
                    for window in sorted(set(natural_windows + equal_windows))}
        reference_window = int(cfg["equal_sample_reference_window_days"])
        reference = eligible[reference_window]
        equal_val_groups = _validation_groups(reference, float(cfg["validation_fraction"]), int(main_cfg["seed"]) + fold)
        equal_capacity = len(reference[~reference.track_group_id.astype(str).isin(equal_val_groups)])
        tasks = [("natural", window, seed) for window in natural_windows for seed in repeat_seeds]
        tasks += [("equal_sample", window, seed) for window in equal_windows for seed in repeat_seeds]
        for design, window, seed in tasks:
            path = _task_path(root, design, window, fold, seed)
            if cfg.get("resume", True) and path.exists():
                all_rows.extend(json.loads(path.read_text(encoding="utf-8"))); continue
            pool = eligible[window]
            if design == "natural":
                val_groups = _validation_groups(pool, float(cfg["validation_fraction"]), seed)
                val = pool[pool.track_group_id.astype(str).isin(val_groups)].copy()
                train = pool[~pool.track_group_id.astype(str).isin(val_groups)].copy()
                assignment = {"target_rows": len(train), "actual_rows": len(train), "groups": int(train.track_group_id.nunique())}
            else:
                val = reference[reference.track_group_id.astype(str).isin(equal_val_groups)].copy()
                candidates = pool[~pool.track_group_id.astype(str).isin(equal_val_groups)].copy()
                train, assignment = select_stratified_track_groups(
                    candidates, equal_capacity, seed, [float(x) for x in cfg["depth_bins_m"]])
            train = finite_rows(train, features + ["Depth"]); val = finite_rows(val, features + ["Depth"])
            if len(train) < int(main_cfg.get("min_train_rows", 500)) or len(val) < int(main_cfg.get("min_validation_rows", 100)):
                raise RuntimeError(f"insufficient data: {design}/w{window}/fold{fold}/seed{seed}")
            model_root = root / "models" / design / f"w{window:03d}" / f"fold{fold}" / f"seed{seed}"
            prediction, weights, info = _fit_full_sample_convex_blend(
                main_cfg, train, val, test, model_root, seed,
                bool(cfg.get("save_models", False)))
            base = {"design": design, "window_days": window, "fold": fold, "seed": seed,
                    "evaluation_protocol": "single_fixed_spatial_holdout",
                    "fit_rows": len(train) + len(val), "train_rows": len(train),
                    "validation_rows": len(val), "test_rows": len(test),
                    "train_tracks": int(pd.concat([train, val]).track_group_id.nunique()),
                    "median_time_delta_days": float(pd.concat([train, val]).time_delta_days.median()),
                    "p90_time_delta_days": float(pd.concat([train, val]).time_delta_days.quantile(.9)),
                    "season_mismatch_rate": float(pd.concat([train, val]).season_mismatch.mean()),
                    "sample_target_rows": int(assignment["target_rows"]),
                    "sample_actual_rows": int(assignment["actual_rows"]),
                    **{f"weight_{name}": float(weight) for name, weight in zip(info["base_regressors"], weights)}}
            rows = _metric_rows(test, prediction, base)
            write_json(path, rows); all_rows.extend(rows); completed += 1
            pd.DataFrame(all_rows).to_csv(root / "metrics" / "task_metrics.csv", index=False)
            print(f"experiment4 {completed}: {design} window={window} fold={fold} seed={seed} train={len(train)}", flush=True)
            if max_tasks is not None and completed >= max_tasks:
                return {"status": "partial", "completed_tasks": completed, "metrics_rows": len(all_rows)}
    pd.DataFrame(all_rows).to_csv(root / "metrics" / "task_metrics.csv", index=False)
    return {"status": "completed", "completed_tasks": completed, "metrics_rows": len(all_rows)}


def _holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues); adjusted = np.empty(len(pvalues), float); running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[index]); running = max(running, value); adjusted[index] = running
    return adjusted.tolist()


def report(main_cfg: dict[str, Any], cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    root = experiment_root(main_cfg, run_id, cfg)
    files = sorted((root / "metrics" / "tasks").glob("*.json"))
    rows = [row for path in files for row in json.loads(path.read_text(encoding="utf-8"))]
    if not rows: raise RuntimeError("no experiment 4 task metrics; run the experiment first")
    frame = pd.DataFrame(rows); frame.to_csv(root / "metrics" / "task_metrics.csv", index=False)
    metric_cols = ["n", "rmse", "mae", "bias", "r2", "pearson_r2"]
    group = ["design", "window_days", "scope", "region_id", "season"]
    summary = frame.groupby(group, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(x).strip("_") for x in summary.columns.to_flat_index()]
    summary.to_csv(root / "metrics" / "summary_mean_sd.csv", index=False)
    tests = []
    comparison = frame[frame.scope.isin(["overall", "region_season"])].copy()
    for (design, scope, region_id, season), data in comparison.groupby(
            ["design", "scope", "region_id", "season"], dropna=False):
        reference_window = int(cfg["equal_sample_reference_window_days"])
        reference = data[data.window_days.eq(reference_window)].set_index(["fold", "seed"])
        for window in sorted(data.window_days.unique()):
            if int(window) == reference_window: continue
            other = data[data.window_days.eq(window)].set_index(["fold", "seed"])
            joined = reference[["rmse"]].join(other[["rmse"]], lsuffix="_ref", rsuffix="_other", how="inner")
            if len(joined) < 2: continue
            delta = joined.rmse_other - joined.rmse_ref
            try: pvalue = float(wilcoxon(delta).pvalue)
            except ValueError: pvalue = 1.0
            tests.append({"design": design, "comparison_scope": scope, "region_id": region_id,
                          "season": season, "reference_window_days": reference_window,
                          "window_days": int(window), "pairs": len(joined),
                          "rmse_difference_mean": float(delta.mean()),
                          "rmse_difference_median": float(delta.median()), "p_value": pvalue})
    if tests:
        adjusted = _holm([row["p_value"] for row in tests])
        for row, value in zip(tests, adjusted): row["p_holm"] = value
    pd.DataFrame(tests).to_csv(root / "metrics" / "paired_window_tests.csv", index=False)
    distribution = frame[frame.scope.eq("overall")][
        ["design", "window_days", "fold", "seed", "train_rows", "train_tracks",
         "median_time_delta_days", "p90_time_delta_days", "season_mismatch_rate"]].drop_duplicates()
    distribution.to_csv(root / "metrics" / "sample_and_season_diagnostics.csv", index=False)
    master = _load_master(main_cfg, root)
    sample_rows = []
    for window in [int(x) for x in cfg["natural_windows_days"]]:
        eligible = master[master.time_delta_days.le(window)]
        for (region_id, season), group_data in eligible.groupby(["region_id", "track_season"]):
            sample_rows.append({"window_days": window, "region_id": region_id, "season": season,
                                "rows": len(group_data), "tracks": int(group_data.track_group_id.nunique()),
                                "median_time_delta_days": float(group_data.time_delta_days.median()),
                                "p90_time_delta_days": float(group_data.time_delta_days.quantile(.9)),
                                "season_mismatch_rate": float(group_data.season_mismatch.mean())})
    pd.DataFrame(sample_rows).to_csv(root / "metrics" / "sample_distribution_by_region_season.csv", index=False)
    result = {"status": "reported", "task_files": len(files), "metric_rows": len(frame),
              "summary": str(root / "metrics" / "summary_mean_sd.csv"),
              "paired_tests": str(root / "metrics" / "paired_window_tests.csv"),
              "season_diagnostics": str(root / "metrics" / "sample_and_season_diagnostics.csv"),
              "regional_season_distribution": str(root / "metrics" / "sample_distribution_by_region_season.csv")}
    write_json(root / "report_summary.json", result); return result
