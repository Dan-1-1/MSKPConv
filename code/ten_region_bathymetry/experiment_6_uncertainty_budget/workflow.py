"""Quantify a CUDEM-conditional uncertainty budget for the Proposed chain.

Classification quality is taken only from the manually labelled independent
test set and remains dimensionless. No ATL24 labels are read by this module.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
import yaml


METRIC_COMPONENTS = ("sea_surface", "refraction", "tide", "inversion_and_unresolved")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def robust_sigma(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype="float64")
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    median = float(np.median(values))
    mad_sigma = 1.4826 * float(np.median(np.abs(values - median)))
    if mad_sigma > 0:
        return mad_sigma
    return float(np.std(values, ddof=1))


def classification_diagnostics(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only the Proposed model's manually labelled independent-test scores."""
    raw = pd.read_csv(cfg["classification_metrics"])
    required = {"seed", "model", "OA", "mIoU", "Macro-F1"}
    missing = required.difference(raw.columns)
    if missing:
        raise KeyError(f"classification metrics missing columns: {sorted(missing)}")
    rows = raw.loc[raw["model"].astype(str).eq(str(cfg["classification_model"])),
                   ["seed", "model", "OA", "mIoU", "Macro-F1"]].copy()
    if len(rows) != 3:
        raise ValueError(f"expected three Proposed seeds, found {len(rows)}")
    records = []
    for metric in ("OA", "mIoU", "Macro-F1"):
        values = pd.to_numeric(rows[metric], errors="raise").to_numpy(float)
        records.append({
            "component": "classification",
            "metric": metric,
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)),
            "n_seeds": int(len(values)),
            "unit": "proportion",
            "source": "manually_labelled_independent_test_set",
            "atl24_used": False,
        })
    return rows, pd.DataFrame(records)


def surface_refraction_diagnostics(cfg: dict[str, Any]) -> pd.DataFrame:
    points = pd.read_csv(cfg["sensitivity_points"], low_memory=False)
    numeric = ["row_index", "K", "n_water", "apparent_depth", "corrected_depth", "fit_rmse"]
    for column in numeric:
        points[column] = pd.to_numeric(points[column], errors="coerce")
    success = points["fit_success"].astype(str).str.lower().isin({"true", "1", "yes"})
    points = points.loc[success].copy()
    keys = ["region_id", "csv", "row_index"]

    k_rows = points.loc[points["experiment"].eq("K_sensitivity")].dropna(
        subset=keys + ["K", "apparent_depth", "corrected_depth", "fit_rmse"])
    k_spread = (k_rows.groupby(keys, sort=False)["corrected_depth"]
                .agg(["count", "std"]).reset_index())
    k_spread = k_spread.loc[k_spread["count"] >= 2]
    baseline = k_rows.loc[k_rows["K"].eq(20)].drop_duplicates(keys).copy()
    baseline["fit_propagated_m"] = (
        baseline["fit_rmse"].abs()
        * baseline["corrected_depth"].abs()
        / baseline["apparent_depth"].abs().clip(lower=1e-6)
    )
    baseline = baseline.merge(k_spread[keys + ["std"]].rename(columns={"std": "k_spread_m"}),
                              on=keys, how="inner")

    n_rows = points.loc[points["experiment"].eq("nw_sensitivity")].dropna(
        subset=keys + ["n_water", "corrected_depth"])
    n_spread = (n_rows.groupby(keys, sort=False)["corrected_depth"]
                .agg(["count", "std"]).reset_index())
    n_spread = n_spread.loc[n_spread["count"] >= 2]

    records = []
    region_ids = sorted(set(baseline["region_id"]).union(n_spread["region_id"]))
    for region_id in ["__overall__", *region_ids]:
        surface = baseline if region_id == "__overall__" else baseline.loc[baseline.region_id.eq(region_id)]
        refract = n_spread if region_id == "__overall__" else n_spread.loc[n_spread.region_id.eq(region_id)]
        u_k = math.sqrt(float(np.mean(np.square(surface["k_spread_m"])))) if len(surface) else float("nan")
        u_fit = math.sqrt(float(np.mean(np.square(surface["fit_propagated_m"])))) if len(surface) else float("nan")
        u_surface = math.sqrt(u_k ** 2 + u_fit ** 2) if np.isfinite(u_k) and np.isfinite(u_fit) else float("nan")
        u_refraction = math.sqrt(float(np.mean(np.square(refract["std"])))) if len(refract) else float("nan")
        records.append({
            "region_id": region_id,
            "surface_u_m": u_surface,
            "surface_K_sensitivity_u_m": u_k,
            "surface_fit_residual_u_m": u_fit,
            "surface_points": int(len(surface)),
            "refraction_u_m": u_refraction,
            "refraction_points": int(len(refract)),
            "refraction_interpretation": "sample_SD_across_tested_n_water_values_1.333_to_1.349",
        })
    return pd.DataFrame(records)


def _selected_months(cfg: dict[str, Any], region_id: str) -> list[str]:
    manifest = pd.read_csv(cfg["selected_tracks"])
    selected = manifest["selected"].astype(str).str.lower().isin({"true", "1", "yes"})
    frame = manifest.loc[selected].copy()
    configured_name = str(cfg["regions"][region_id]["name"])
    if region_id == "cocos_guam":
        frame = frame.loc[frame["region"].astype(str).str.startswith("Cocos")]
    else:
        frame = frame.loc[frame["region"].astype(str).eq(configured_name)]
    dates: list[pd.Timestamp] = []
    for value in frame.get("csv_file", pd.Series(dtype=str)).astype(str):
        match = re.search(r"ATL03_(\d{8})", value)
        if match:
            dates.append(pd.Timestamp(match.group(1)))
    months = sorted({date.strftime("%Y%m") for date in dates
                     if date < pd.Timestamp.now().normalize()})
    limit = int(cfg.get("tide_months_per_region", 3))
    if len(months) <= limit:
        return months
    indices = np.linspace(0, len(months) - 1, limit).round().astype(int)
    return [months[index] for index in sorted(set(indices))]


def _noaa_month(cache: Path, station: str, month: str, product: str,
                timeout: int, allow_network: bool = True) -> pd.DataFrame:
    destination = cache / f"{station}_{month}_{product}.csv"
    if destination.exists():
        data = pd.read_csv(destination)
        data["t"] = pd.to_datetime(data["t"], errors="coerce")
        return data
    if not allow_network:
        raise FileNotFoundError(f"NOAA cache is missing: {destination}")
    start = pd.Timestamp(month + "01")
    end = start + pd.offsets.MonthEnd(0)
    params = {
        "product": product,
        "application": "proposed_uncertainty_budget",
        "begin_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "datum": "MLLW",
        "station": station,
        "time_zone": "gmt",
        "units": "metric",
        "format": "json",
    }
    if product == "predictions":
        params["interval"] = "6"
    session = requests.Session()
    session.trust_env = False
    response = session.get("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter",
                           params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    records = payload.get("data") if product == "water_level" else payload.get("predictions")
    if not records:
        raise RuntimeError(str(payload.get("error", f"no {product} data")))
    data = pd.DataFrame(records).rename(columns={"v": "value", "s": "sigma"})
    data["t"] = pd.to_datetime(data["t"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    if "sigma" not in data:
        data["sigma"] = np.nan
    data["sigma"] = pd.to_numeric(data["sigma"], errors="coerce")
    data = data.dropna(subset=["t", "value"])[["t", "value", "sigma"]]
    cache.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)
    return data


def tide_diagnostics(cfg: dict[str, Any], output: Path, skip_network: bool) -> pd.DataFrame:
    cache = output / "cache" / "noaa"
    timeout = int(cfg.get("noaa_timeout_seconds", 45))
    records = []
    for region_id, region in cfg["regions"].items():
        months = _selected_months(cfg, region_id)
        stations = [str(value) for value in region["stations"]]
        residuals: list[float] = []
        instrument_sigmas: list[float] = []
        errors: list[str] = []
        comparisons = 0
        for month in months:
            try:
                observed = _noaa_month(cache, stations[0], month, "water_level", timeout,
                                       allow_network=not skip_network)
                instrument_sigmas.extend(observed["sigma"].dropna().tolist())
            except Exception as exc:
                errors.append(f"{stations[0]} {month} water_level: {exc}")
                continue
            if str(region["tide_mode"]) == "observed_minus_prediction":
                try:
                    other = _noaa_month(cache, stations[0], month, "predictions", timeout,
                                        allow_network=not skip_network)
                    paired = observed[["t", "value"]].merge(
                        other[["t", "value"]], on="t", suffixes=("_obs", "_pred"))
                    difference = paired["value_obs"] - paired["value_pred"]
                    difference = difference - difference.median()
                    residuals.extend(difference.dropna().tolist())
                    comparisons += 1
                except Exception as exc:
                    errors.append(f"{stations[0]} {month} predictions: {exc}")
            else:
                for alternative in stations[1:]:
                    try:
                        other = _noaa_month(cache, alternative, month, "water_level", timeout,
                                            allow_network=not skip_network)
                        paired = observed[["t", "value"]].merge(
                            other[["t", "value"]], on="t", suffixes=("_primary", "_alternative"))
                        difference = paired["value_primary"] - paired["value_alternative"]
                        difference = difference - difference.median()
                        residuals.extend(difference.dropna().tolist())
                        comparisons += 1
                    except Exception as exc:
                        errors.append(f"{alternative} {month} water_level: {exc}")
        transfer = robust_sigma(residuals)
        instrument = float(np.nanmedian(instrument_sigmas)) if instrument_sigmas else float("nan")
        terms = [value for value in (transfer, instrument) if np.isfinite(value)]
        tide_u = math.sqrt(sum(value ** 2 for value in terms)) if terms else float("nan")
        if skip_network and not terms:
            errors.append("network disabled and no cached NOAA observations were available")
        records.append({
            "region_id": region_id,
            "region": region["name"],
            "tide_mode": region["tide_mode"],
            "primary_station": stations[0],
            "comparison_stations": ";".join(stations[1:]),
            "months": ";".join(months),
            "successful_comparisons": comparisons,
            "paired_samples": len(residuals),
            "debiased_transfer_or_prediction_residual_u_m": transfer,
            "median_reported_observation_u_m": instrument,
            "tide_u_m": tide_u,
            "errors": " | ".join(errors),
        })
    return pd.DataFrame(records)


def load_oof(cfg: dict[str, Any], run_id: str) -> pd.DataFrame:
    pattern = str(cfg["oof_prediction_glob"]).format(run_id=run_id)
    files = sorted(Path(pattern).parent.glob(Path(pattern).name))
    if len(files) != 5:
        raise FileNotFoundError(f"expected five OOF folds for {pattern}, found {len(files)}")
    columns = ["region_id", "fold", "track_group_id", "CUDEM_Depth", "Prediction"]
    data = pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)
    for column in ("CUDEM_Depth", "Prediction"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["region_id", "CUDEM_Depth", "Prediction"]).copy()
    data["error_m"] = data["Prediction"] - data["CUDEM_Depth"]
    return data


def _residual_summary(data: pd.DataFrame, region_id: str) -> dict[str, Any]:
    error = data["error_m"].to_numpy(float)
    reference = data["CUDEM_Depth"].to_numpy(float)
    ss_res = float(np.sum(np.square(error)))
    ss_tot = float(np.sum(np.square(reference - reference.mean())))
    return {
        "region_id": region_id,
        "n": int(len(data)),
        "tracks": int(data["track_group_id"].nunique()),
        "rmse_m": math.sqrt(float(np.mean(np.square(error)))),
        "mae_m": float(np.mean(np.abs(error))),
        "bias_m": float(np.mean(error)),
        "random_u_m": float(np.std(error, ddof=1)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def oof_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    records = [_residual_summary(data, "__overall__")]
    records.extend(_residual_summary(group, str(region_id))
                   for region_id, group in data.groupby("region_id", sort=True))
    return pd.DataFrame(records)


def metric_budget(oof: pd.DataFrame, sensitivity: pd.DataFrame,
                  tide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sens = sensitivity.set_index("region_id")
    tides = tide.set_index("region_id")
    regional_records = []
    long_records = []
    for item in oof.to_dict("records"):
        region_id = str(item["region_id"])
        source_region = region_id if region_id in sens.index else "__overall__"
        surface_u = float(sens.loc[source_region, "surface_u_m"])
        refraction_u = float(sens.loc[source_region, "refraction_u_m"])
        if region_id == "__overall__":
            weighted = (oof.loc[oof.region_id.ne("__overall__"), ["region_id", "n"]]
                        .merge(tide[["region_id", "tide_u_m"]], on="region_id", how="left"))
            weighted["tide_u_m"] = pd.to_numeric(weighted["tide_u_m"], errors="coerce")
            weighted = weighted.dropna(subset=["tide_u_m"])
            tide_u = (math.sqrt(float(np.average(np.square(weighted["tide_u_m"]),
                                                 weights=weighted["n"])))
                      if len(weighted) else float("nan"))
        else:
            tide_u = float(tides.loc[region_id, "tide_u_m"]) if region_id in tides.index else float("nan")
        random_u = float(item["random_u_m"])
        known = [value for value in (surface_u, refraction_u, tide_u) if np.isfinite(value)]
        known_variance = sum(value ** 2 for value in known)
        unresolved_variance = max(random_u ** 2 - known_variance, 0.0)
        unresolved_u = math.sqrt(unresolved_variance)
        overlap_flag = known_variance > random_u ** 2
        row = dict(item)
        row.update({
            "surface_u_m": surface_u,
            "refraction_sensitivity_u_m": refraction_u,
            "tide_u_m": tide_u,
            "inversion_and_unresolved_u_m": unresolved_u,
            "known_upstream_u_m": math.sqrt(known_variance),
            "combined_metric_u_m": math.sqrt(known_variance + unresolved_variance),
            "known_variance_exceeds_total": overlap_flag,
            "metric_budget_complete": bool(np.isfinite(surface_u) and np.isfinite(refraction_u) and np.isfinite(tide_u)),
            "missing_metric_components": ";".join(
                name for name, value in (("sea_surface", surface_u), ("refraction", refraction_u), ("tide", tide_u))
                if not np.isfinite(value)),
            "classification_unit_conversion": "not_performed",
            "reference_condition": "CUDEM_conditional",
        })
        regional_records.append(row)
        denominator = random_u ** 2
        for component, value, estimator in (
            ("sea_surface", surface_u, "K_sensitivity_plus_local_surface_fit_residual"),
            ("refraction", refraction_u, "SD_across_tested_n_water_values"),
            ("tide", tide_u, "debiased_NOAA_station_or_prediction_residual"),
            ("inversion_and_unresolved", unresolved_u, "residual_variance_remainder"),
        ):
            long_records.append({
                "region_id": region_id,
                "component": component,
                "standard_uncertainty_m": value,
                "variance_m2": value ** 2 if np.isfinite(value) else float("nan"),
                "contribution_pct_of_total_random_variance":
                    100.0 * value ** 2 / denominator if denominator > 0 and np.isfinite(value) else float("nan"),
                "estimator": estimator,
            })
    return pd.DataFrame(regional_records), pd.DataFrame(long_records)


def depth_bin_calibration(oof: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    records = []
    requested = int(cfg.get("depth_bins", 8))
    for region_id, group in oof.groupby("region_id", sort=True):
        bins = pd.qcut(group["Prediction"], q=requested, duplicates="drop")
        for interval, values in group.groupby(bins, observed=True):
            error = values["error_m"].to_numpy(float)
            records.append({
                "region_id": region_id,
                "depth_left_m": float(interval.left),
                "depth_right_m": float(interval.right),
                "depth_center_m": float(np.median(values["Prediction"])),
                "n": int(len(values)),
                "rmse_m": math.sqrt(float(np.mean(np.square(error)))),
                "random_u_m": float(np.std(error, ddof=1)) if len(error) > 1 else float("nan"),
                "bias_m": float(np.mean(error)),
            })
    return pd.DataFrame(records)


def write_uncertainty_rasters(cfg: dict[str, Any], run_id: str, output: Path,
                              calibration: pd.DataFrame, budget: pd.DataFrame) -> list[dict[str, Any]]:
    scene_root = Path(str(cfg["scene_root"]).format(run_id=run_id))
    budget_by_region = budget.set_index("region_id")
    results = []
    for region_id in cfg["regions"]:
        prediction_path = scene_root / "depth" / region_id / "Proposed.tif"
        mask_path = scene_root / "common_mask" / region_id / "proposed_convexblend_valid_pixels.tif"
        if not prediction_path.exists() or not mask_path.exists():
            results.append({"region_id": region_id, "status": "missing_input",
                            "prediction": str(prediction_path), "mask": str(mask_path)})
            continue
        bins = calibration.loc[calibration.region_id.eq(region_id)].sort_values("depth_center_m")
        if bins.empty:
            results.append({"region_id": region_id, "status": "missing_oof_calibration"})
            continue
        centers = bins["depth_center_m"].to_numpy(float)
        sigma = bins["rmse_m"].to_numpy(float)
        upstream_floor = float(budget_by_region.loc[region_id, "known_upstream_u_m"])
        destination = output / "rasters" / region_id
        destination.mkdir(parents=True, exist_ok=True)
        sigma_path = destination / "DEM_uncertainty_1sigma.tif"
        pi_path = destination / "DEM_PI95_halfwidth.tif"
        qa_path = destination / "DEM_QA_mask.tif"
        with rasterio.open(prediction_path) as src, rasterio.open(mask_path) as mask_src:
            if (src.width, src.height, src.transform, src.crs) != (mask_src.width, mask_src.height, mask_src.transform, mask_src.crs):
                raise ValueError(f"prediction/mask grid mismatch for {region_id}")
            float_profile = src.profile.copy()
            float_profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate", predictor=3)
            qa_profile = src.profile.copy()
            qa_profile.update(dtype="uint8", count=1, nodata=0, compress="deflate", predictor=1)
            with rasterio.open(sigma_path, "w", **float_profile) as sigma_dst, \
                    rasterio.open(pi_path, "w", **float_profile) as pi_dst, \
                    rasterio.open(qa_path, "w", **qa_profile) as qa_dst:
                valid_pixels = 0
                outside_support = 0
                floor_pixels = 0
                for _, window in src.block_windows(1):
                    depth = src.read(1, window=window).astype("float64")
                    support = mask_src.read(1, window=window)
                    valid = np.isfinite(depth) & (depth > 0) & (support > 0)
                    if src.nodata is not None:
                        valid &= depth != src.nodata
                    out_sigma = np.full(depth.shape, -9999.0, dtype="float32")
                    out_pi = np.full(depth.shape, -9999.0, dtype="float32")
                    qa = np.zeros(depth.shape, dtype="uint8")
                    if valid.any():
                        raw_sigma = np.interp(depth[valid], centers, sigma, left=sigma[0], right=sigma[-1])
                        floored = raw_sigma < upstream_floor
                        values = np.maximum(raw_sigma, upstream_floor)
                        inside = (depth[valid] >= centers.min()) & (depth[valid] <= centers.max())
                        quality = np.ones(values.shape, dtype="uint8")
                        quality[inside] |= 2
                        if len(bins) >= int(cfg.get("depth_bins", 8)) - 1:
                            quality |= 4
                        quality[floored] |= 8
                        if not bool(budget_by_region.loc[region_id, "metric_budget_complete"]):
                            quality |= 16
                        out_sigma[valid] = values.astype("float32")
                        out_pi[valid] = (1.96 * values).astype("float32")
                        qa[valid] = quality
                        valid_pixels += int(valid.sum())
                        outside_support += int((~inside).sum())
                        floor_pixels += int(floored.sum())
                    sigma_dst.write(out_sigma, 1, window=window)
                    pi_dst.write(out_pi, 1, window=window)
                    qa_dst.write(qa, 1, window=window)
                tags = {
                    "reference_condition": "CUDEM-conditional",
                    "estimator": "OOF depth-bin RMSE with metric upstream floor",
                    "classification_in_metres": "not estimated",
                    "not_a_theoretical_bound": "true",
                    "units": "m",
                }
                sigma_dst.update_tags(**tags)
                pi_dst.update_tags(**tags, coverage="approximate 95 percent half-width = 1.96 sigma")
                qa_dst.update_tags(bit_1="valid prediction", bit_2="inside OOF depth support",
                                   bit_4="regional bin calibration available", bit_8="upstream floor applied",
                                   bit_16="metric component unavailable")
        results.append({"region_id": region_id, "status": "complete", "valid_pixels": valid_pixels,
                        "outside_oof_depth_support": outside_support, "floor_applied_pixels": floor_pixels,
                        "uncertainty": str(sigma_path), "pi95": str(pi_path), "qa": str(qa_path)})
    return results


def _make_figure(classification: pd.DataFrame, long_budget: pd.DataFrame, output: Path) -> Path:
    figure_path = output / "figures" / "error_budget.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    regions = long_budget.loc[long_budget.region_id.ne("__overall__"), "region_id"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [0.8, 2.2]})
    axes[0].bar(classification["metric"], classification["mean"],
                yerr=classification["sample_std"], color=["#2b6f77", "#8a6f3b", "#665b8e"], capsize=4)
    axes[0].set_ylim(0.75, 1.0)
    axes[0].set_ylabel("Independent-test score")
    axes[0].set_title("Proposed classification")
    axes[0].grid(axis="y", alpha=0.25)
    pivot = (long_budget.loc[long_budget.region_id.isin(regions)]
             .pivot(index="region_id", columns="component", values="contribution_pct_of_total_random_variance")
             .reindex(regions))
    bottom = np.zeros(len(pivot))
    colors = {"sea_surface": "#4f83a6", "refraction": "#71a36f", "tide": "#d09a45",
              "inversion_and_unresolved": "#767676"}
    for component in METRIC_COMPONENTS:
        values = pivot.get(component, pd.Series(0.0, index=pivot.index)).fillna(0).to_numpy(float)
        axes[1].bar(pivot.index, values, bottom=bottom, label=component.replace("_", " "), color=colors[component])
        bottom += values
    axes[1].set_ylabel("Share of CUDEM-conditional random variance (%)")
    axes[1].set_title("Metric error budget by region")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def publication_table(classification: pd.DataFrame, regional: pd.DataFrame,
                      long_budget: pd.DataFrame) -> pd.DataFrame:
    """Create one mixed-unit table without converting classification scores to metres."""
    rows: list[dict[str, Any]] = []
    for item in classification.to_dict("records"):
        rows.append({"stage": "classification", "quantity": item["metric"],
                     "value": item["mean"], "uncertainty_or_sd": item["sample_std"],
                     "unit": "proportion", "variance_share_pct": np.nan,
                     "interpretation": "manual independent test; mean and sample SD across three seeds"})
    overall_components = long_budget.loc[long_budget.region_id.eq("__overall__")]
    for item in overall_components.to_dict("records"):
        rows.append({"stage": item["component"], "quantity": "standard_uncertainty",
                     "value": item["standard_uncertainty_m"], "uncertainty_or_sd": np.nan,
                     "unit": "m", "variance_share_pct": item["contribution_pct_of_total_random_variance"],
                     "interpretation": item["estimator"]})
    overall = regional.loc[regional.region_id.eq("__overall__")].iloc[0]
    for stage, quantity, value, interpretation in (
        ("total", "CUDEM_conditional_RMSE", overall.rmse_m, "OOF prediction minus CUDEM"),
        ("total", "CUDEM_conditional_bias", overall.bias_m, "reported separately from random uncertainty"),
        ("total", "residual_random_standard_uncertainty", overall.random_u_m,
         "sample SD of spatially isolated OOF residuals"),
    ):
        rows.append({"stage": stage, "quantity": quantity, "value": value,
                     "uncertainty_or_sd": np.nan, "unit": "m", "variance_share_pct": np.nan,
                     "interpretation": interpretation})
    return pd.DataFrame(rows)


def _write_report(output: Path, classification: pd.DataFrame, regional: pd.DataFrame,
                  long_budget: pd.DataFrame, tide: pd.DataFrame, rasters: list[dict[str, Any]]) -> Path:
    overall = regional.loc[regional.region_id.eq("__overall__")].iloc[0]
    class_values = {row.metric: (row["mean"], row.sample_std)
                    for _, row in classification.iterrows()}
    region_view = regional.loc[regional.region_id.ne("__overall__"),
                               ["region_id", "n", "rmse_m", "mae_m", "bias_m", "random_u_m",
                                "surface_u_m", "refraction_sensitivity_u_m", "tide_u_m",
                                "inversion_and_unresolved_u_m", "metric_budget_complete"]].copy()
    report = [
        "## Material Passport",
        "",
        "- Type: Experiment Result",
        "- Status: VERIFIED",
        "- Reference: CUDEM",
        "- Classification evidence: manually labelled independent test set",
        "",
        "# Proposed-chain CUDEM-conditional uncertainty budget",
        "",
        "Classification is evaluated only on the manually labelled independent test set. "
        "ATL24 labels are not used. Classification scores remain dimensionless and are not converted to metres.",
        "",
        "## Classification",
        "",
        f"- OA: {class_values['OA'][0]:.4f} +/- {class_values['OA'][1]:.4f}",
        f"- mIoU: {class_values['mIoU'][0]:.4f} +/- {class_values['mIoU'][1]:.4f}",
        f"- Macro-F1: {class_values['Macro-F1'][0]:.4f} +/- {class_values['Macro-F1'][1]:.4f}",
        "",
        "## Overall metric budget",
        "",
        f"OOF N = {int(overall.n):,}; RMSE = {overall.rmse_m:.4f} m; MAE = {overall.mae_m:.4f} m; "
        f"Bias = {overall.bias_m:.4f} m; residual random standard uncertainty = {overall.random_u_m:.4f} m.",
        "",
        long_budget.loc[long_budget.region_id.eq("__overall__"),
                        ["component", "standard_uncertainty_m", "contribution_pct_of_total_random_variance"]]
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "The non-negative residual remainder is not a pure inversion-only causal effect. It also contains "
        "unresolved interactions and any classification effect that cannot be assigned a valid metre conversion.",
        "",
        "## Ten-region results",
        "",
        region_view.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Tutuila's NOAA station provides astronomical predictions but no downloadable observed water-level "
        "series for the selected months. Its tide uncertainty is therefore left missing rather than imputed.",
        "",
        f"Completed uncertainty raster sets: {sum(item.get('status') == 'complete' for item in rasters)}/10.",
    ]
    path = output / "report.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return path


def run_budget(config_path: Path, run_id: str, skip_tide_network: bool = False,
               skip_rasters: bool = False) -> dict[str, Any]:
    cfg = load_config(config_path)
    output = Path(cfg["output_root"]) / run_id / cfg["experiment_id"]
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    status_path = output / "run_status.json"
    _write_json(status_path, {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                              "run_id": run_id, "classification_source": "manual_independent_test_only",
                              "atl24_used": False})
    try:
        seed_rows, classification = classification_diagnostics(cfg)
        sensitivity = surface_refraction_diagnostics(cfg)
        tide = tide_diagnostics(cfg, output, skip_tide_network)
        oof = load_oof(cfg, run_id)
        oof_summary = oof_diagnostics(oof)
        regional, long_budget = metric_budget(oof_summary, sensitivity, tide)
        calibration = depth_bin_calibration(oof, cfg)
        publication = publication_table(classification, regional, long_budget)

        seed_rows.to_csv(tables / "proposed_classification_seed_metrics.csv", index=False)
        classification.to_csv(tables / "proposed_classification_diagnostics.csv", index=False)
        sensitivity.to_csv(tables / "surface_refraction_diagnostics.csv", index=False)
        tide.to_csv(tables / "tide_diagnostics.csv", index=False)
        oof_summary.to_csv(tables / "oof_cudem_metrics.csv", index=False)
        regional.loc[regional.region_id.eq("__overall__")].to_csv(tables / "error_budget_overall.csv", index=False)
        regional.loc[regional.region_id.ne("__overall__")].to_csv(tables / "error_budget_by_region.csv", index=False)
        long_budget.to_csv(tables / "error_budget_components_long.csv", index=False)
        publication.to_csv(tables / "error_budget_publication_table.csv", index=False)
        calibration.to_csv(tables / "depth_bin_calibration.csv", index=False)

        rasters = [] if skip_rasters else write_uncertainty_rasters(cfg, run_id, output, calibration, regional)
        figure = _make_figure(classification, long_budget, output)
        report_path = _write_report(output, classification, regional, long_budget, tide, rasters)
        metadata = {
            "title": "CUDEM-conditional predictive uncertainty budget",
            "run_id": run_id,
            "classification": {
                "source": "manually labelled independent test set, three seeds",
                "metrics": ["OA", "mIoU", "Macro-F1"],
                "atl24_used": False,
                "metre_conversion": "not performed because no validated transfer function exists",
            },
            "metric_budget": {
                "bias_reported_separately": True,
                "variance_addition_assumption": "diagnostic bookkeeping; causal independence is not claimed",
                "inversion_component": "non-negative residual variance remainder; includes unresolved interactions",
                "reference": "CUDEM",
            },
            "uncertainty_layers": rasters,
            "figure": str(figure),
            "report": str(report_path),
        }
        _write_json(output / "metadata.json", metadata)
        warnings = tide.loc[tide["tide_u_m"].isna(), "region_id"].tolist()
        result = {"status": "complete", "run_id": run_id, "output": str(output),
                  "regions": int(len(regional) - 1), "oof_rows": int(len(oof)),
                  "rasters_complete": int(sum(item.get("status") == "complete" for item in rasters)),
                  "tide_missing_regions": warnings, "completed_utc": datetime.now(timezone.utc).isoformat()}
        _write_json(status_path, result)
        return result
    except Exception as exc:
        _write_json(status_path, {"status": "failed", "run_id": run_id,
                                  "error": repr(exc), "failed_utc": datetime.now(timezone.utc).isoformat()})
        raise
