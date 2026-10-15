"""Full representative-scene inversion on identical shallow-water CUDEM support."""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import friedmanchisquare, wilcoxon

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from experiment_1_main.workflow import (  # noqa: E402
    FEATURE_FILES,
    cudem_rasters,
    load_config,
    metrics,
    model_run_dir,
    regions_from_config,
    scene_inventory,
    write_json,
)
from experiment_2_model_selection.workflow import _read_sentinel2_stack  # noqa: E402


def load_scene_config(path: Path | None = None) -> dict[str, Any]:
    path = path or HERE / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def experiment_root(main_cfg: dict[str, Any], scene_cfg: dict[str, Any], run_id: str) -> Path:
    root = Path(main_cfg["output_root"]) / run_id / "scene_evaluation" / str(scene_cfg["experiment_id"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def raster_root(main_cfg: dict[str, Any], scene_cfg: dict[str, Any], run_id: str) -> Path:
    base = Path(scene_cfg.get("raster_output_root", main_cfg.get("model_output_root", main_cfg["output_root"])))
    root = base / run_id / str(scene_cfg["experiment_id"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def _configured_regions(main_cfg: dict[str, Any], scene_cfg: dict[str, Any]):
    wanted = set(str(value) for value in scene_cfg["regions"])
    return [region for region in regions_from_config(main_cfg) if region.id in wanted]


def _representative_scene(region):
    scenes, _ = scene_inventory(region)
    return next((item for item in scenes if item["scene_id"] == region.representative_scene), None)


def _model_files(main_cfg: dict[str, Any], run_id: str, method: str) -> list[Path]:
    root = model_run_dir(main_cfg, run_id) / "models" / "final_full" / method
    blend_path = root / "ConvexBlend" / "final.json"
    if not blend_path.exists():
        return [blend_path]
    blend = json.loads(blend_path.read_text(encoding="utf-8"))
    files = [blend_path]
    for name in blend["base_regressors"]:
        suffix = ".cbm" if name == "CatBoost" else ".joblib"
        files.append(root / name / f"final{suffix}")
    return files


def qc(main_cfg: dict[str, Any], scene_cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    import rasterio

    root = experiment_root(main_cfg, scene_cfg, run_id)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    methods = [str(value) for value in scene_cfg["methods"]]
    for region in _configured_regions(main_cfg, scene_cfg):
        scene = _representative_scene(region)
        item: dict[str, Any] = {
            "region_id": region.id,
            "representative_scene": region.representative_scene,
            "scene_exists": scene is not None,
        }
        if scene is None:
            errors.append(f"missing representative scene: {region.id}/{region.representative_scene}")
            checks.append(item)
            continue
        scene_dir = Path(scene["scene_dir"])
        required = [scene_dir / name for name in FEATURE_FILES.values()] + [scene_dir / "cloud_mask.tif"]
        raw_stack = scene_dir / "B02_B03_B04_B08_Stack.tif"
        if not raw_stack.exists():
            required.extend(scene_dir / f"{band}.tif" for band in ("B02", "B03", "B04", "B08"))
        missing = [str(path) for path in required if not path.exists()]
        cudem = cudem_rasters(region)
        if not cudem:
            missing.append(str(region.cudem))
        item.update({"scene_dir": str(scene_dir), "cudem_rasters": len(cudem), "missing": missing})
        if not missing:
            with rasterio.open(scene_dir / FEATURE_FILES["B02"]) as src:
                item.update({"height": src.height, "width": src.width, "crs": str(src.crs)})
            for path in required[1:]:
                with rasterio.open(path) as src:
                    if src.height != item["height"] or src.width != item["width"] or str(src.crs) != item["crs"]:
                        missing.append(f"unaligned raster: {path}")
        errors.extend(missing)
        checks.append(item)
    for method in methods:
        missing = [str(path) for path in _model_files(main_cfg, run_id, method) if not path.exists()]
        if missing:
            errors.extend(missing)
        checks.append({"method": method, "model_variant": "final_full", "missing_models": missing})
    result = {
        "status": "PASS" if not errors else "FAIL",
        "run_id": run_id,
        "regions": len(_configured_regions(main_cfg, scene_cfg)),
        "methods": len(methods),
        "errors": errors,
        "checks": checks,
    }
    write_json(root / "qc" / "summary.json", result)
    if errors:
        raise RuntimeError("scene-evaluation QC failed: " + "; ".join(errors[:8]))
    return result


def _write_float_raster(path: Path, values: np.ndarray, profile: dict[str, Any], tags: dict[str, Any]) -> None:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    output_profile = profile.copy()
    output_profile.update(
        dtype="float32", count=1, nodata=-9999.0, compress="DEFLATE",
        tiled=True, blockxsize=256, blockysize=256,
    )
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(np.where(np.isfinite(values), values, -9999.0).astype("float32"), 1)
        dst.update_tags(**{key: str(value) for key, value in tags.items()})


def _write_mask_raster(path: Path, values: np.ndarray, profile: dict[str, Any], tags: dict[str, Any]) -> None:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    output_profile = profile.copy()
    output_profile.update(
        dtype="uint8", count=1, nodata=0, compress="DEFLATE",
        tiled=True, blockxsize=256, blockysize=256,
    )
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(values.astype("uint8"), 1)
        dst.update_tags(**{key: str(value) for key, value in tags.items()})


def _read_float_raster(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    import rasterio

    with rasterio.open(path) as src:
        values = src.read(1, masked=True).filled(np.nan).astype("float32")
        return values, src.profile.copy()


def _aligned_cudem_depth(region, scene_path: Path, output_path: Path, resampling_name: str) -> np.ndarray:
    """Reproject CUDEM elevation to the Sentinel-2 grid and convert z to depth.

    CUDEM products use elevation, so underwater cells are negative.  The
    conversion is explicitly depth=-elevation; this remains correct in scenes
    whose land pixels dominate the raster distribution.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    if output_path.exists():
        values, _ = _read_float_raster(output_path)
        return values
    with rasterio.open(scene_path) as ref:
        profile = ref.profile.copy()
        elevation = np.full((ref.height, ref.width), np.nan, dtype="float32")
        for index, source_path in enumerate(cudem_rasters(region)):
            with rasterio.open(source_path) as src:
                reproject(
                    source=rasterio.band(src, 1), destination=elevation,
                    src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                    dst_transform=ref.transform, dst_crs=ref.crs, dst_nodata=np.nan,
                    resampling=getattr(Resampling, str(resampling_name)),
                    init_dest_nodata=index == 0,
                )
    depth = -elevation
    # Retain the scene's complete underwater CUDEM range. Land elevations
    # become negative depths and are excluded; deep cells are never clipped.
    depth[~np.isfinite(depth) | (depth < 0.0)] = np.nan
    scene_max_depth = float(np.nanmax(depth)) if np.isfinite(depth).any() else np.nan
    _write_float_raster(
        output_path, depth, profile,
        {"region_id": region.id, "reference": "CUDEM", "conversion": "depth=-elevation",
         "depth_domain": "full_scene_cudem_bathymetry", "scene_max_depth_m": scene_max_depth},
    )
    return depth


def _predict_method(main_cfg: dict[str, Any], scene_cfg: dict[str, Any], run_id: str,
                    method: str, features: np.ndarray, raw_stack: np.ndarray,
                    cloud: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from catboost import CatBoostRegressor
    from joblib import load
    from experiment_1_main.workflow import _predict_cnn_raster_points

    method_root = model_run_dir(main_cfg, run_id) / "models" / "final_full" / method
    blend = json.loads((method_root / "ConvexBlend" / "final.json").read_text(encoding="utf-8"))
    prediction = np.zeros(len(features), dtype="float64")
    for name, weight in zip(blend["base_regressors"], blend["weights"]):
        weight = float(weight)
        if abs(weight) <= 1e-12:
            continue
        if name == "CatBoost":
            model = CatBoostRegressor(); model.load_model(str(method_root / name / "final.cbm"))
        else:
            model = load(method_root / name / "final.joblib")
        if name == "LightweightCNN":
            base_prediction = _predict_cnn_raster_points(
                model, raw_stack, cloud, valid,
                int(main_cfg["lightweight_cnn"].get("image_patch_size", 11)),
                int(scene_cfg.get("cnn_chunk_size", 20000)),
            )
        else:
            base_prediction = np.asarray(model.predict(features), dtype="float64")
        prediction += weight * base_prediction
        del model, base_prediction; gc.collect()
    return prediction


def _sufficient_statistics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - reference
    return {
        "sum_reference": float(reference.sum(dtype="float64")),
        "sum_reference_sq": float(np.square(reference, dtype="float64").sum(dtype="float64")),
        "sum_prediction": float(prediction.sum(dtype="float64")),
        "sum_prediction_sq": float(np.square(prediction, dtype="float64").sum(dtype="float64")),
        "sum_reference_prediction": float((reference.astype("float64") * prediction.astype("float64")).sum()),
        "sum_error": float(error.sum(dtype="float64")),
        "sum_sq_error": float(np.square(error, dtype="float64").sum(dtype="float64")),
        "sum_abs_error": float(np.abs(error).sum(dtype="float64")),
    }


def _region_metrics(region_id: str, scene_id: str, method: str, reference: np.ndarray,
                    prediction: np.ndarray, support: str) -> dict[str, Any]:
    result = metrics(reference, prediction)
    result.update(_sufficient_statistics(reference, prediction))
    result.update({
        "region_id": region_id, "scene_id": scene_id, "method": method,
        "reference": "CUDEM", "support": support,
    })
    return result


def _load_region_features(main_cfg: dict[str, Any], region, scene_dir: Path):
    import rasterio

    feature_arrays = []
    profile = None
    for feature in main_cfg["features"]:
        with rasterio.open(scene_dir / FEATURE_FILES[str(feature)]) as src:
            if profile is None:
                profile = src.profile.copy()
            feature_arrays.append(src.read(1, out_dtype="float32"))
    cube = np.stack(feature_arrays, axis=-1)
    with rasterio.open(scene_dir / "cloud_mask.tif") as src:
        cloud = src.read(1)
    valid = np.isfinite(cube).all(axis=-1) & (cube[:, :, :4] > 0).all(axis=-1) & (cloud == 0)
    raw_stack = _read_sentinel2_stack(scene_dir)
    return cube, cloud, valid, raw_stack, profile


def run(main_cfg: dict[str, Any], scene_cfg: dict[str, Any], run_id: str,
        max_regions: int | None = None) -> dict[str, Any]:
    output = experiment_root(main_cfg, scene_cfg, run_id)
    rasters = raster_root(main_cfg, scene_cfg, run_id)
    methods = [str(value) for value in scene_cfg["methods"]]
    rows: list[dict[str, Any]] = []
    completed_regions = 0
    for region in _configured_regions(main_cfg, scene_cfg):
        metrics_path = output / "metrics" / "regions" / f"{region.id}.json"
        if bool(scene_cfg.get("resume", True)) and metrics_path.exists():
            item = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.extend(item["metrics"])
            completed_regions += 1
            print(f"scene evaluation reused {completed_regions}: {region.id}", flush=True)
            if max_regions is not None and completed_regions >= max_regions:
                break
            continue
        scene = _representative_scene(region)
        if scene is None:
            raise FileNotFoundError(f"representative scene missing for {region.id}")
        scene_dir = Path(scene["scene_dir"])
        cube, cloud, valid, raw_stack, profile = _load_region_features(main_cfg, region, scene_dir)
        features = cube[valid]
        del cube
        gc.collect()
        method_paths: dict[str, Path] = {}
        for method in methods:
            depth_path = rasters / "depth" / region.id / f"{method}.tif"
            method_paths[method] = depth_path
            if bool(scene_cfg.get("resume", True)) and depth_path.exists():
                print(f"scene map reused: {region.id}/{method}", flush=True)
                continue
            prediction = _predict_method(
                main_cfg, scene_cfg, run_id, method, features, raw_stack, cloud, valid
            )
            depth = np.full(valid.shape, np.nan, dtype="float32")
            depth[valid] = prediction.astype("float32")
            _write_float_raster(depth_path, depth, profile, {
                "region_id": region.id, "scene_id": region.representative_scene,
                "method": method, "regressor": "ConvexBlend", "model_variant": "final_full",
                "run_id": run_id,
            })
            del prediction, depth
            gc.collect()
            print(f"scene map completed: {region.id}/{method}", flush=True)
        cudem_path = rasters / "reference" / region.id / "CUDEM_depth.tif"
        cudem = _aligned_cudem_depth(
            region, scene_dir / FEATURE_FILES["B02"], cudem_path,
            str(scene_cfg.get("cudem_resampling", "bilinear")),
        )
        prediction_arrays: dict[str, np.ndarray] = {}
        common = np.isfinite(cudem)
        for method, path in method_paths.items():
            values, _ = _read_float_raster(path)
            prediction_arrays[method] = values
            common &= np.isfinite(values)
        if not common.any():
            raise RuntimeError(f"no valid Proposed-ConvexBlend CUDEM support for {region.id}")
        support = str(scene_cfg["support"])
        if scene_cfg.get("save_common_mask", True):
            _write_mask_raster(
                rasters / "common_mask" / region.id / f"{support}.tif",
                common, profile,
                {"region_id": region.id, "scene_id": region.representative_scene,
                 "support": support, "valid_pixels": int(common.sum())},
            )
        region_rows = []
        reference = cudem[common].astype("float64")
        for method in methods:
            row = _region_metrics(
                region.id, region.representative_scene, method,
                reference, prediction_arrays[method][common].astype("float64"),
                support,
            )
            region_rows.append(row)
            rows.append(row)
        write_json(metrics_path, {
            "region_id": region.id, "scene_id": region.representative_scene,
            "common_pixels": int(common.sum()), "metrics": region_rows,
        })
        pd.DataFrame(rows).to_csv(output / "metrics" / "region_metrics.csv", index=False)
        completed_regions += 1
        print(f"scene evaluation {completed_regions}: {region.id} common={int(common.sum())}", flush=True)
        del features, raw_stack, cloud, valid, cudem, prediction_arrays, reference, common
        gc.collect()
        if max_regions is not None and completed_regions >= max_regions:
            break
    pd.DataFrame(rows).to_csv(output / "metrics" / "region_metrics.csv", index=False)
    status = {"status": "completed" if completed_regions == len(_configured_regions(main_cfg, scene_cfg)) else "partial",
              "regions": completed_regions, "expected_regions": len(_configured_regions(main_cfg, scene_cfg)),
              "methods": len(methods)}
    write_json(output / "run_status.json", status)
    return status


def _pooled_metrics(group: pd.DataFrame) -> dict[str, float]:
    n = int(group["n"].sum())
    sy = float(group["sum_reference"].sum())
    sy2 = float(group["sum_reference_sq"].sum())
    sp = float(group["sum_prediction"].sum())
    sp2 = float(group["sum_prediction_sq"].sum())
    syp = float(group["sum_reference_prediction"].sum())
    sse = float(group["sum_sq_error"].sum())
    sst = sy2 - sy * sy / n
    var_y = sy2 - sy * sy / n
    var_p = sp2 - sp * sp / n
    covariance = syp - sy * sp / n
    pearson_r2 = covariance * covariance / (var_y * var_p) if var_y > 0 and var_p > 0 else np.nan
    return {
        "n": n, "rmse": float(np.sqrt(sse / n)),
        "mae": float(group["sum_abs_error"].sum() / n),
        "bias": float(group["sum_error"].sum() / n),
        "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
        "pearson_r2": float(pearson_r2),
    }


def _make_figures(main_cfg: dict[str, Any], scene_cfg: dict[str, Any], run_id: str,
                  region_metrics: pd.DataFrame, output: Path, rasters: Path) -> None:
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.enums import Resampling

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    methods = [str(value) for value in scene_cfg["methods"]]
    matrix = region_metrics.pivot(index="region_id", columns="method", values="rmse").reindex(
        index=[str(value) for value in scene_cfg["regions"]], columns=methods
    )
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iat[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if value > np.nanmedian(matrix.to_numpy()) else "black")
    ax.set_title("Proposed–ConvexBlend representative-scene RMSE against CUDEM")
    fig.colorbar(image, ax=ax, label="RMSE (m)")
    fig.savefig(figures / "regional_rmse_heatmap.png", dpi=220)
    plt.close(fig)

    region_map = {region.id: region for region in _configured_regions(main_cfg, scene_cfg)}
    panels = ["CUDEM", *methods]
    for region_id in matrix.index:
        sources = [rasters / "reference" / region_id / "CUDEM_depth.tif"] + [
            rasters / "depth" / region_id / f"{method}.tif" for method in methods
        ]
        if not all(path.exists() for path in sources):
            continue
        common_path = rasters / "common_mask" / region_id / f"{scene_cfg['support']}.tif"
        with rasterio.open(common_path) as src:
            scale = max(1.0, np.sqrt((src.height * src.width) / 800_000.0))
            out_shape = (max(1, int(src.height / scale)), max(1, int(src.width / scale)))
            common = src.read(1, out_shape=out_shape, resampling=Resampling.nearest).astype(bool)
        arrays = []
        for path in sources:
            with rasterio.open(path) as src:
                values = src.read(1, out_shape=out_shape, masked=True, resampling=Resampling.bilinear).filled(np.nan)
                values[~common] = np.nan
                arrays.append(values)
        ncols = min(4, len(panels))
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                                 constrained_layout=True, squeeze=False)
        last = None
        scene_max_depth = float(np.nanmax(arrays[0]))
        for ax, label, values in zip(axes.flat, panels, arrays):
            last = ax.imshow(values, cmap="viridis_r", vmin=0, vmax=scene_max_depth)
            ax.set_title(label)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        name = region_map[region_id].name if region_id in region_map else region_id
        fig.suptitle(f"{name}: Proposed–ConvexBlend depth and CUDEM reference")
        fig.colorbar(last, ax=axes, label="Depth (m)", shrink=0.8)
        fig.savefig(figures / f"{region_id}_depth_comparison.png", dpi=180)
        plt.close(fig)


def report(main_cfg: dict[str, Any], scene_cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    output = experiment_root(main_cfg, scene_cfg, run_id)
    rasters = raster_root(main_cfg, scene_cfg, run_id)
    path = output / "metrics" / "region_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    expected = len(scene_cfg["regions"]) * len(scene_cfg["methods"])
    if len(data) != expected:
        raise RuntimeError(f"incomplete scene metrics: {len(data)} != {expected}")
    count_matrix = data.pivot(index="region_id", columns="method", values="n")
    if count_matrix.isna().any().any() or not count_matrix.nunique(axis=1).eq(1).all():
        raise RuntimeError("scene comparison does not use identical per-region support")
    metrics_to_report = ["rmse", "mae", "bias", "r2", "pearson_r2"]
    macro = data.groupby("method", as_index=False)[metrics_to_report].agg(["mean", "std"]).reset_index()
    macro.columns = ["_".join(value).strip("_") for value in macro.columns.to_flat_index()]
    macro.to_csv(output / "metrics" / "macro_region_summary.csv", index=False)
    pooled_rows = []
    for method, group in data.groupby("method"):
        pooled_rows.append({"method": method, **_pooled_metrics(group)})
    pooled = pd.DataFrame(pooled_rows).sort_values("rmse")
    pooled.to_csv(output / "metrics" / "pixel_pooled_summary.csv", index=False)
    pivot = data.pivot(index="region_id", columns="method", values="rmse")
    statistics: dict[str, Any] = {}
    if len(pivot) >= 3 and pivot.shape[1] >= 3:
        test = friedmanchisquare(*[pivot[column].to_numpy(float) for column in pivot.columns])
        statistics["friedman_rmse"] = {"statistic": float(test.statistic), "p_value": float(test.pvalue)}
        comparisons = []
        for method in scene_cfg["methods"]:
            method = str(method)
            if method == "Proposed":
                continue
            diff = pivot[method] - pivot["Proposed"]
            test = wilcoxon(diff.to_numpy(float), alternative="two-sided", method="auto")
            comparisons.append({"comparison": f"{method} vs Proposed", "n_regions": len(diff),
                                "mean_rmse_difference": float(diff.mean()),
                                "statistic": float(test.statistic), "p_value": float(test.pvalue)})
        comparison_df = pd.DataFrame(comparisons)
        comparison_df["p_value_holm"] = np.maximum.accumulate(
            np.minimum(1.0, comparison_df.sort_values("p_value")["p_value"].to_numpy() *
                       np.arange(len(comparison_df), 0, -1))
        )[np.argsort(np.argsort(comparison_df["p_value"].to_numpy()))]
        comparison_df.to_csv(output / "metrics" / "significance_vs_proposed.csv", index=False)
        statistics["wilcoxon_vs_proposed"] = comparison_df.to_dict("records")
    ranks = pivot.rank(axis=1, ascending=True, method="average")
    rank_summary = ranks.mean().sort_values().rename("mean_rank").reset_index()
    rank_summary.to_csv(output / "metrics" / "mean_ranks.csv", index=False)
    _make_figures(main_cfg, scene_cfg, run_id, data, output, rasters)
    result = {
        "status": "reported", "run_id": run_id,
        "reference": "CUDEM", "support": str(scene_cfg["support"]),
        "regions": len(scene_cfg["regions"]), "methods": len(scene_cfg["methods"]),
        "identical_support_within_region": True,
        "macro_region_summary": macro.to_dict("records"),
        "pixel_pooled_summary": pooled.to_dict("records"),
        "mean_ranks": rank_summary.to_dict("records"),
        "statistics": statistics,
        "raster_root": str(rasters),
    }
    write_json(output / "report_summary.json", result)
    write_json(output / "run_status.json", {"status": "reported", "regions": len(scene_cfg["regions"]), "methods": len(scene_cfg["methods"])})
    return result
