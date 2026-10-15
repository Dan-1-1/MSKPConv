"""Track-level, spatially isolated regional few-shot adaptation.

This module intentionally reads the strict split products produced by the main
ten-region workflow.  It never re-splits individual photons, so every reported
adaptation result has a fixed held-out target track set and a configurable
spatial exclusion (500 m by default, matching the main experiment).
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
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from experiment_1_main.workflow import (  # noqa: E402
    depth_domain_mask,
    cudem_rasters,
    finite_rows,
    load_config,
    metrics,
    regions_from_config,
    scene_inventory,
    stable_order,
    write_json,
)


METRIC_COLUMNS = ("n", "rmse", "mae", "median_ae", "bias", "abs_bias", "r2", "pearson_r2")


def load_adaptation_config(path: Path | None = None) -> dict[str, Any]:
    path = path or (HERE / "config.yaml")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def adaptation_root(main_cfg: dict[str, Any], main_run_id: str, adapt_cfg: dict[str, Any]) -> Path:
    root = Path(main_cfg["output_root"]) / main_run_id / "regional_adaptation" / str(adapt_cfg["experiment_id"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def _split_path(main_cfg: dict[str, Any], run_id: str, region_id: str, method: str) -> Path:
    return Path(main_cfg["output_root"]) / run_id / "splits" / region_id / f"{method}.parquet"


def _group_rows(df: pd.DataFrame, groups: Iterable[str]) -> pd.DataFrame:
    groups = set(groups)
    return df[df["track_group_id"].astype(str).isin(groups)].copy()


def _depth_hist(depth: np.ndarray, bins: list[float]) -> np.ndarray:
    if len(depth) == 0:
        return np.zeros(len(bins) + 1, dtype=float)
    return np.histogram(depth, bins=[-np.inf, *bins[1:], np.inf])[0].astype(float)


def select_depth_balanced_groups(
    df: pd.DataFrame,
    proportion: float,
    seed: int,
    depth_bins: list[float],
    depth_balance_weight: float = 0.5,
    randomness_weight: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select whole tracks close to a target count while preserving depth range.

    The greedy objective is deterministic for a given seed.  It is deliberately
    group based: no track is ever split between adaptation and target test data.
    """
    if not 0 <= proportion <= 1:
        raise ValueError("proportion must be between 0 and 1")
    if df.empty or proportion <= 0:
        return df.iloc[0:0].copy(), {"target_rows": 0, "actual_rows": 0, "groups": [], "actual_proportion": 0.0}
    grouped = []
    for gid, g in df.groupby(df["track_group_id"].astype(str), sort=False):
        grouped.append((gid, g, len(g), _depth_hist(g["Depth"].to_numpy(float), depth_bins)))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(grouped))
    shuffled = [grouped[int(i)] for i in order]
    random_priority = {item[0]: rank / max(len(shuffled) - 1, 1) for rank, item in enumerate(shuffled)}
    target = int(round(len(df) * proportion))
    if proportion > 0:
        target = max(1, target)
    total_hist = _depth_hist(df["Depth"].to_numpy(float), depth_bins)
    total_hist = total_hist / max(total_hist.sum(), 1.0)
    selected: list[tuple[str, pd.DataFrame, int, np.ndarray]] = []
    selected_rows = 0
    selected_hist = np.zeros_like(total_hist)
    remaining = list(shuffled)
    while remaining and selected_rows < target:
        best_idx, best_score = None, None
        for i, (gid, g, n, hist) in enumerate(remaining):
            proposed_rows = selected_rows + n
            count_err = abs(proposed_rows - target) / max(target, 1)
            prop_hist = (selected_hist + hist) / max(proposed_rows, 1)
            depth_err = float(np.abs(prop_hist - total_hist).sum()) / 2.0
            overshoot = max(0, proposed_rows - target) / max(target, 1)
            score = ((1.0 - depth_balance_weight) * count_err + depth_balance_weight * depth_err
                     + 0.25 * overshoot + float(randomness_weight) * random_priority[gid])
            tie = random_priority[gid]
            candidate = (score, tie)
            if best_score is None or candidate < best_score:
                best_idx, best_score = i, candidate
        gid, g, n, hist = remaining.pop(int(best_idx))
        selected.append((gid, g, n, hist)); selected_rows += n; selected_hist += hist
    groups = [x[0] for x in selected]
    result = _group_rows(df, groups)
    return result, {"target_rows": int(target), "actual_rows": int(len(result)), "groups": groups,
                    "actual_proportion": float(len(result) / len(df)) if len(df) else 0.0}


def select_nested_track_groups(df: pd.DataFrame, proportions: list[float], seed: int,
                               depth_bins: list[float], spatial_grid_m: float) -> dict[float, tuple[pd.DataFrame, dict[str, Any]]]:
    """Build one deterministic, spatial/depth/season-balanced nested track sequence."""
    work = df.copy()
    depth = pd.cut(work.Depth, bins=[-np.inf, *depth_bins[1:], np.inf], labels=False).astype(str)
    if {"x", "y"}.issubset(work.columns):
        spatial = (np.floor(work.x / spatial_grid_m).astype(int).astype(str) + "_" +
                   np.floor(work.y / spatial_grid_m).astype(int).astype(str))
    else:
        spatial = (work.Longitude.round(2).astype(str) + "_" + work.Latitude.round(2).astype(str))
    month = work.scene_id.astype(str).str.extract(r"_(\d{6})_")[0].str[-2:].fillna("00")
    season = ((pd.to_numeric(month, errors="coerce").fillna(1).astype(int) - 1) // 3).astype(str)
    work["_stratum"] = spatial + "|d" + depth + "|q" + season
    labels = sorted(work._stratum.unique()); label_index = {value: i for i, value in enumerate(labels)}
    target_hist = np.zeros(len(labels), dtype=float)
    for label, count in work._stratum.value_counts().items(): target_hist[label_index[label]] = count
    target_hist /= max(target_hist.sum(), 1.0)
    groups = []
    for gid, group in work.groupby(work.track_group_id.astype(str), sort=False):
        hist = np.zeros(len(labels), dtype=float)
        for label, count in group._stratum.value_counts().items(): hist[label_index[label]] = count
        groups.append((gid, len(group), hist))
    remaining = sorted(groups, key=lambda item: stable_order(item[0], seed))
    selected: list[str] = []; selected_rows = 0; selected_hist = np.zeros(len(labels), dtype=float)
    outputs = {}
    for proportion in sorted(set(float(value) for value in proportions)):
        target_rows = int(round(len(work) * proportion))
        if proportion > 0: target_rows = max(1, target_rows)
        while remaining and selected_rows < target_rows:
            scored = []
            for index, (gid, size, hist) in enumerate(remaining):
                proposed_rows = selected_rows + size
                composition = (selected_hist + hist) / max(proposed_rows, 1)
                score = (abs(proposed_rows - target_rows) / max(target_rows, 1)
                         + 0.75 * np.abs(composition - target_hist).sum() / 2
                         + 0.25 * max(0, proposed_rows - target_rows) / max(target_rows, 1))
                scored.append((score, stable_order(gid, seed), index))
            _, _, index = min(scored)
            gid, size, hist = remaining.pop(index); selected.append(gid)
            selected_rows += size; selected_hist += hist
        chosen = _group_rows(df, selected)
        outputs[proportion] = (chosen, {"target_rows": target_rows, "actual_rows": len(chosen),
                                        "groups": list(selected),
                                        "actual_proportion": float(len(chosen) / len(df)) if len(df) else 0.0,
                                        "nested": True})
    return outputs


def spatial_block_bootstrap(reference: np.ndarray, prediction: np.ndarray, common: np.ndarray,
                            transform: Any, block_m: float, repeats: int, confidence: float,
                            seed: int) -> dict[str, float]:
    """Return spatial-block bootstrap CIs without treating adjacent pixels as independent."""
    rows, cols = np.where(common); error = prediction - reference
    block_rows = max(1, int(round(block_m / abs(float(transform.e)))))
    block_cols = max(1, int(round(block_m / abs(float(transform.a)))))
    block_id = (rows // block_rows) * (int(np.ceil(common.shape[1] / block_cols))) + cols // block_cols
    _, inverse = np.unique(block_id, return_inverse=True); count = np.bincount(inverse).astype(float)
    sums = {
        "y": np.bincount(inverse, weights=reference), "y2": np.bincount(inverse, weights=reference ** 2),
        "p": np.bincount(inverse, weights=prediction), "p2": np.bincount(inverse, weights=prediction ** 2),
        "yp": np.bincount(inverse, weights=reference * prediction),
        "e": np.bincount(inverse, weights=error), "e2": np.bincount(inverse, weights=error ** 2),
        "ae": np.bincount(inverse, weights=np.abs(error)),
    }
    rng = np.random.default_rng(seed); samples = {name: [] for name in ("rmse", "mae", "bias", "r2", "pearson_r2")}
    for _ in range(int(repeats)):
        chosen = rng.integers(0, len(count), len(count)); n = count[chosen].sum()
        total = {key: value[chosen].sum() for key, value in sums.items()}
        sst = total["y2"] - total["y"] ** 2 / n
        vary = sst; varp = total["p2"] - total["p"] ** 2 / n
        cov = total["yp"] - total["y"] * total["p"] / n
        samples["rmse"].append(float(np.sqrt(total["e2"] / n)))
        samples["mae"].append(float(total["ae"] / n)); samples["bias"].append(float(total["e"] / n))
        samples["r2"].append(float(1 - total["e2"] / sst) if sst > 0 else np.nan)
        samples["pearson_r2"].append(float(cov ** 2 / (vary * varp)) if vary > 0 and varp > 0 else np.nan)
    alpha = (1.0 - float(confidence)) / 2.0; result = {"bootstrap_blocks": int(len(count)), "bootstrap_repeats": int(repeats)}
    for name, values in samples.items():
        result[f"{name}_ci_low"] = float(np.nanquantile(values, alpha))
        result[f"{name}_ci_high"] = float(np.nanquantile(values, 1 - alpha))
    return result


def _cap_by_groups(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= int(max_rows):
        return df.copy()
    groups = [(str(g), len(gdf)) for g, gdf in df.groupby("track_group_id", sort=False)]
    groups.sort(key=lambda x: (-x[1], stable_order(x[0], int(seed))))
    chosen, n = [], 0
    for gid, size in groups:
        if n + size > int(max_rows) and chosen:
            continue
        chosen.append(gid); n += size
        if n >= int(max_rows):
            break
    return _group_rows(df, chosen)


def _check_spatial_split(df: pd.DataFrame, split_col: str, buffer_m: float) -> dict[str, Any]:
    train = df[df[split_col].eq("train")]
    test = df[df[split_col].eq("test")]
    overlap = set(train.track_group_id.astype(str)) & set(test.track_group_id.astype(str))
    if overlap:
        raise ValueError(f"track overlap in {split_col}: {sorted(overlap)[:3]}")
    if train.empty or test.empty or not {"x", "y"}.issubset(df.columns):
        return {"min_train_test_distance_m": None, "group_overlap": 0}
    tree = cKDTree(test[["x", "y"]].to_numpy(float))
    dist, _ = tree.query(train[["x", "y"]].to_numpy(float), k=1)
    return {"min_train_test_distance_m": float(np.min(dist)), "group_overlap": 0,
            "buffer_ok": bool(float(np.min(dist)) >= float(buffer_m))}


def qc(main_cfg: dict[str, Any], adapt_cfg: dict[str, Any], main_run_id: str) -> dict[str, Any]:
    root = adaptation_root(main_cfg, main_run_id, adapt_cfg)
    regions = regions_from_config(main_cfg)
    target_ids = [str(x) for x in adapt_cfg["target_regions"]]
    methods = [str(x) for x in adapt_cfg["methods"]]
    checks, errors = [], []
    for rid in target_ids:
        for method in methods:
            p = _split_path(main_cfg, main_run_id, rid, method)
            item = {"region_id": rid, "method": method, "path": str(p), "exists": p.exists()}
            if not p.exists():
                errors.append(f"missing {p}"); checks.append(item); continue
            d = pd.read_parquet(p); item.update({"rows": len(d), "groups": int(d.track_group_id.nunique())})
            region = next(value for value in regions if value.id == rid)
            scenes, _ = scene_inventory(region)
            scene = next((value for value in scenes if value["scene_id"] == region.representative_scene), None)
            item["representative_scene"] = region.representative_scene
            item["scene_exists"] = scene is not None
            item["cudem_rasters"] = len(cudem_rasters(region))
            if scene is None: errors.append(f"missing representative scene: {rid}/{region.representative_scene}")
            if not cudem_rasters(region): errors.append(f"missing CUDEM: {region.cudem}")
            checks.append(item)
    available = {r.id for r in regions}
    errors.extend([f"unknown target region {x}" for x in target_ids if x not in available])
    result = {"status": "PASS" if not errors else "FAIL", "main_run_id": main_run_id,
              "checks": checks, "errors": errors}
    write_json(root / "qc" / "summary.json", result)
    if errors:
        raise RuntimeError("regional adaptation QC failed: " + "; ".join(errors[:5]))
    return result


def _all_region_data(main_cfg: dict[str, Any], run_id: str, method: str) -> dict[str, pd.DataFrame]:
    features = list(main_cfg["features"]); result = {}
    for region in regions_from_config(main_cfg):
        frame = pd.read_parquet(_split_path(main_cfg, run_id, region.id, method)).copy()
        frame["adapt_group_id"] = region.id + "|" + frame.track_group_id.astype(str)
        result[region.id] = finite_rows(frame, features + ["Depth"])
    return result


def _validation_groups_full(data: pd.DataFrame, fraction: float, seed: int) -> set[str]:
    sizes = data.groupby("adapt_group_id").size().to_dict()
    target = max(1, int(round(len(data) * float(fraction))))
    ordered = sorted(sizes, key=lambda value: stable_order(value, int(seed)))
    chosen, rows = [], 0
    for group in ordered:
        chosen.append(group); rows += int(sizes[group])
        if rows >= target: break
    return set(chosen)


def _fit_and_predict_full_scene(main_cfg: dict[str, Any], data: pd.DataFrame,
                                scene_features: np.ndarray, raw_stack: np.ndarray,
                                cloud: np.ndarray, valid: np.ndarray,
                                seed: int, validation_fraction: float,
                                cnn_chunk_size: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Tune on held-out tracks, refit every base on all rows, then predict the scene."""
    from catboost import CatBoostRegressor
    from experiment_1_main.workflow import _predict_cnn_raster_points
    from experiment_2_model_selection.workflow import _fit_convex_weights, _models, extract_sentinel2_patches

    features = list(main_cfg["features"])
    val_groups = _validation_groups_full(data, validation_fraction, seed)
    train = data[~data.adapt_group_id.isin(val_groups)].copy()
    validation = data[data.adapt_group_id.isin(val_groups)].copy()
    if len(train) < int(main_cfg.get("min_train_rows", 500)) or len(validation) < int(main_cfg.get("min_validation_rows", 100)):
        raise RuntimeError(f"insufficient train/validation rows: {len(train)}/{len(validation)}")
    base_names = [str(value) for value in main_cfg["convex_blend"]["base_regressors"]]
    tuning_models = _models(main_cfg, seed); val_predictions = []; tuning = {}
    point_patches = None
    if "LightweightCNN" in base_names:
        point_patches = extract_sentinel2_patches(
            pd.concat([train, validation], ignore_index=True), main_cfg,
            int(main_cfg["lightweight_cnn"].get("image_patch_size", 11)))
    for name in base_names:
        model = tuning_models[name]
        if name == "LightweightCNN":
            model.fit(point_patches[:len(train)], train.Depth.to_numpy(float),
                      eval_set=(point_patches[len(train):], validation.Depth.to_numpy(float)))
            val_pred = model.predict(point_patches[len(train):]); tuning["cnn_best_epoch"] = int(model.best_epoch_)
        elif name == "CatBoost":
            model.fit(train[features], train.Depth, eval_set=(validation[features], validation.Depth), use_best_model=True)
            val_pred = model.predict(validation[features]); tuning["catboost_best_iteration"] = int(model.get_best_iteration())
        else:
            model.fit(train[features], train.Depth); val_pred = model.predict(validation[features])
        val_predictions.append(np.asarray(val_pred, dtype="float64"))
        del model; gc.collect()
    weights = _fit_convex_weights(validation.Depth.to_numpy(float), np.column_stack(val_predictions))
    final_models = _models(main_cfg, seed); scene_predictions = []
    full_patches = None
    if "LightweightCNN" in base_names:
        full_patches = extract_sentinel2_patches(data, main_cfg, int(main_cfg["lightweight_cnn"].get("image_patch_size", 11)))
    for name in base_names:
        model = final_models[name]
        if name == "LightweightCNN":
            model.fit_full(full_patches, data.Depth.to_numpy(float), max(1, int(tuning.get("cnn_best_epoch", 1))))
            pred = _predict_cnn_raster_points(model, raw_stack, cloud, valid,
                                              int(main_cfg["lightweight_cnn"].get("image_patch_size", 11)), cnn_chunk_size)
        elif name == "CatBoost":
            params = dict(main_cfg["catboost"]); params["iterations"] = max(1, int(tuning.get("catboost_best_iteration", 0)) + 1)
            params.pop("early_stopping_rounds", None)
            model = CatBoostRegressor(**params, verbose=False, eval_metric="RMSE")
            model.fit(data[features], data.Depth); pred = model.predict(scene_features)
        else:
            if name == "MLP": model.set_params(mlpregressor__early_stopping=False)
            model.fit(data[features], data.Depth); pred = model.predict(scene_features)
        scene_predictions.append(np.asarray(pred, dtype="float64")); del model; gc.collect()
    info = {"base_regressors": base_names, "weights": weights.tolist(), "fit_rows": len(data),
            "train_rows_for_weight_tuning": len(train), "validation_rows_for_weight_tuning": len(validation), **tuning}
    return np.column_stack(scene_predictions) @ weights, weights, info


def _load_source_data(main_cfg: dict[str, Any], main_run_id: str, target_id: str, method: str,
                      fold: int, adapt_cfg: dict[str, Any], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts, val_parts = [], []
    for region in regions_from_config(main_cfg):
        if region.id == target_id:
            continue
        p = _split_path(main_cfg, main_run_id, region.id, method)
        if not p.exists():
            continue
        d = pd.read_parquet(p); col = f"split_fold_{int(adapt_cfg['source_validation_fold'])}"
        if col not in d:
            continue
        train_parts.append(_cap_by_groups(d[d[col].eq("train")], adapt_cfg.get("max_source_train_rows_per_region"), seed))
        val_parts.append(_cap_by_groups(d[d[col].eq("validation")], adapt_cfg.get("max_source_validation_rows_per_region"), seed + 17))
    if not train_parts or not val_parts:
        raise RuntimeError("source train/validation data are incomplete")
    return pd.concat(train_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True)


def _task_path(root: Path, target: str, method: str, proportion: float, seed: int) -> Path:
    ptxt = f"p{int(round(proportion * 100)):03d}"
    return root / "metrics" / f"{target}__{method}__{ptxt}__seed{seed}.json"


def _final_base_predictions(main_cfg: dict[str, Any], main_run_id: str, method: str,
                            frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    from catboost import CatBoostRegressor
    from joblib import load
    from experiment_2_model_selection.workflow import extract_sentinel2_patches

    features = list(main_cfg["features"])
    root = model_run_dir(main_cfg, main_run_id) / "models" / "final_full" / method
    info = json.loads((root / "ConvexBlend" / "final.json").read_text(encoding="utf-8"))
    predictions = []
    for name in info["base_regressors"]:
        if name == "CatBoost":
            model = CatBoostRegressor(); model.load_model(str(root / name / "final.cbm"))
        else:
            model = load(root / name / "final.joblib")
        if name == "LightweightCNN":
            patches = extract_sentinel2_patches(
                frame, main_cfg, int(main_cfg["lightweight_cnn"].get("image_patch_size", 11)))
            pred = model.predict(patches)
        else:
            pred = model.predict(frame[features])
        predictions.append(np.asarray(pred, dtype="float64"))
    return np.column_stack(predictions), info


def run(main_cfg: dict[str, Any], adapt_cfg: dict[str, Any], main_run_id: str,
        max_tasks: int | None = None, targets: list[str] | None = None,
        methods: list[str] | None = None,
        proportions: list[float] | None = None,
        seeds: list[int] | None = None) -> dict[str, Any]:
    from scene_evaluation.workflow import (
        FEATURE_FILES, _aligned_cudem_depth, _load_region_features, _write_float_raster,
    )

    root = adaptation_root(main_cfg, main_run_id, adapt_cfg)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    target_ids = targets or [str(x) for x in adapt_cfg["target_regions"]]
    method_ids = methods or [str(x) for x in adapt_cfg["methods"]]
    prop_ids = proportions or [float(x) for x in adapt_cfg["proportions"]]
    seed_ids = seeds or [int(value) for value in adapt_cfg["selection_seeds"]]
    region_map = {region.id: region for region in regions_from_config(main_cfg)}
    all_results, task_count = [], 0
    for method in method_ids:
        region_data = _all_region_data(main_cfg, main_run_id, method)
        for target in target_ids:
            target_region = region_map[target]
            source = pd.concat([value for rid, value in region_data.items() if rid != target], ignore_index=True)
            pool = region_data[target]
            scenes, _ = scene_inventory(target_region)
            scene = next((value for value in scenes if value["scene_id"] == target_region.representative_scene), None)
            if scene is None: raise FileNotFoundError(f"missing representative scene: {target}")
            scene_dir = Path(scene["scene_dir"])
            cube, cloud, valid, raw_stack, profile = _load_region_features(main_cfg, target_region, scene_dir)
            scene_features = cube[valid]; del cube; gc.collect()
            reference_path = root / "reference" / target / "CUDEM_depth.tif"
            cudem = _aligned_cudem_depth(target_region, scene_dir / FEATURE_FILES["B02"], reference_path,
                                         str(adapt_cfg.get("cudem_resampling", "bilinear")))
            common = valid & np.isfinite(cudem)
            reference = cudem[common].astype("float64")
            if reference.size == 0: raise RuntimeError(f"no full-scene CUDEM support: {target}")
            nested_assignments = select_nested_track_groups(
                pool, prop_ids, seed_ids[0], [float(value) for value in adapt_cfg["depth_bins_m"]],
                float(adapt_cfg.get("selection_spatial_grid_m", 1000.0)))
            for proportion in prop_ids:
                for seed in seed_ids:
                    out_json = _task_path(root, target, method, proportion, seed)
                    if adapt_cfg.get("resume", True) and out_json.exists():
                        all_results.append(json.loads(out_json.read_text(encoding="utf-8"))); continue
                    selected, assignment = nested_assignments[float(proportion)]
                    fit_data = pd.concat([source, selected], ignore_index=True)
                    prediction_valid, weights, model_info = _fit_and_predict_full_scene(
                        main_cfg, fit_data, scene_features, raw_stack, cloud, valid, seed,
                        float(adapt_cfg["validation_fraction"]), int(adapt_cfg.get("cnn_chunk_size", 20000)))
                    prediction = np.full(valid.shape, np.nan, dtype="float32")
                    prediction[valid] = prediction_valid.astype("float32")
                    pred_common = prediction[common].astype("float64")
                    mm = metrics(reference, pred_common); mm["abs_bias"] = abs(float(mm["bias"]))
                    uncertainty = spatial_block_bootstrap(
                        reference, pred_common, common, profile["transform"],
                        float(adapt_cfg.get("bootstrap_block_m", 500.0)),
                        int(adapt_cfg.get("bootstrap_repeats", 200)),
                        float(adapt_cfg.get("bootstrap_confidence", 0.95)), seed)
                    record = {
                        "status": "OK", "target_region": target, "scene_id": target_region.representative_scene,
                        "method": method, "regressor": "ConvexBlend",
                        "adaptation_strategy": adapt_cfg["adaptation_strategy"], "primary_reference": "CUDEM",
                        "test_support": adapt_cfg["test_support"], "nominal_proportion": proportion,
                        "selection_seed": seed, "source_regions": 9, "source_rows": len(source),
                        "target_pool_rows": len(pool), "target_adaptation_rows": int(assignment["actual_rows"]),
                        "target_actual_proportion": assignment["actual_proportion"],
                        "target_adaptation_tracks": len(assignment["groups"]),
                        "full_scene_prediction_pixels": int(valid.sum()), "full_scene_cudem_pixels": int(common.sum()),
                        "base_regressors": model_info["base_regressors"], "weights": weights.tolist(),
                        "model_fit_rows": model_info["fit_rows"],
                        "weight_validation_rows": model_info["validation_rows_for_weight_tuning"], **mm, **uncertainty,
                    }
                    write_json(root / "assignments" / f"{target}__p{int(round(proportion * 100)):03d}__seed{seed}.json",
                               {"target_region": target, "method": method,
                                "selection_seed": seed, "nominal_proportion": proportion, **assignment})
                    if adapt_cfg.get("save_scene_maps", False):
                        map_path = root / "scene_maps" / target / f"p{int(round(proportion * 100)):03d}__seed{seed}.tif"
                        _write_float_raster(map_path, prediction, profile, record)
                    write_json(out_json, record); all_results.append(record); task_count += 1
                    print(f"adapt retrain {task_count}: {target}/{method}/p{proportion:.2f}/seed{seed} cudem={int(common.sum())}", flush=True)
                    if max_tasks is not None and task_count >= max_tasks:
                        df = pd.DataFrame(all_results); df.to_csv(root / "metrics" / "adaptation_metrics.csv", index=False)
                        return {"tasks": task_count, "metrics": df.to_dict("records")}
            del scene_features, raw_stack, cloud, valid, cudem, common, reference
            gc.collect()
    df = pd.DataFrame(all_results); df.to_csv(root / "metrics" / "adaptation_metrics.csv", index=False)
    expected = len(target_ids) * len(method_ids) * len(prop_ids) * len(seed_ids)
    write_json(root / "run_status.json", {"status": "completed", "tasks": task_count, "expected_tasks": expected})
    return {"tasks": task_count, "metrics": df.to_dict("records")}


def report(main_cfg: dict[str, Any], adapt_cfg: dict[str, Any], main_run_id: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    root = adaptation_root(main_cfg, main_run_id, adapt_cfg)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    p = root / "metrics" / "adaptation_metrics.csv"
    if not p.exists():
        files = sorted((root / "metrics").glob("*.json")); rows = [json.loads(x.read_text(encoding="utf-8")) for x in files]
        pd.DataFrame(rows).to_csv(p, index=False)
    df = pd.read_csv(p)
    value_cols = ["rmse", "mae", "median_ae", "bias", "abs_bias", "r2", "pearson_r2",
                  "rmse_ci_low", "rmse_ci_high", "mae_ci_low", "mae_ci_high",
                  "bias_ci_low", "bias_ci_high", "r2_ci_low", "r2_ci_high",
                  "target_actual_proportion", "target_adaptation_tracks"]
    group_cols = ["target_region", "method", "scene_id", "nominal_proportion"]
    summary = df.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(value).strip("_") for value in summary.columns.to_flat_index()]
    summary.to_csv(root / "metrics" / "summary.csv", index=False)
    endpoints = summary[summary["nominal_proportion"].isin([0.0, 1.0])].copy()
    endpoints.to_csv(root / "metrics" / "endpoint_comparison.csv", index=False)
    for target, g in summary.groupby("target_region"):
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True)
        x = g["nominal_proportion"] * 100
        for ax, metric, label in zip(axes, ["rmse", "mae", "r2", "bias"], ["RMSE (m)", "MAE (m)", "R²", "Bias (m)"]):
            mean = g[f"{metric}_mean"]
            lower = mean - g[f"{metric}_ci_low_mean"]
            upper = g[f"{metric}_ci_high_mean"] - mean
            ax.errorbar(x, mean, yerr=np.vstack([lower.clip(lower=0), upper.clip(lower=0)]),
                        marker="o", linewidth=1.7, capsize=2)
            ax.set_xlabel("Target adaptation samples (%)"); ax.set_ylabel(label); ax.grid(alpha=0.25)
        fig.suptitle(f"{target}: leave-one-region-out full-scene adaptation")
        fig.savefig(root / "figures" / f"{target}_adaptation.png", dpi=220); plt.close(fig)
    write_json(root / "run_status.json", {"status": "reported", "rows": int(len(df))})
    return {"rows": len(df), "summary": str(root / "metrics" / "summary.csv")}
