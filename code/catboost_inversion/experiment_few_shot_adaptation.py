"""E5: few-shot regional adaptation with repeated sampling.

Training:
  other 3 regions all samples + target region samples at a given proportion.
  Each region/proportion setting is repeated with multiple random seeds.

Testing:
  target-region test CSV + full-scene CUDEM external-consistency validation.

Outputs:
  {region}/Catboost/e5_few_shot/
      results.json
      repeat_details.csv
      {region}_few_shot_{XXX}pct_summary.json
      repeat_{YY}/
          {region}_few_shot_{XXX}pct_seed{SEED}_icesat2_scatter.png
          {region}_few_shot_{XXX}pct_seed{SEED}_cudem_scatter.png
          {region}_few_shot_{XXX}pct_seed{SEED}_predicted_depth.tif

  {DATA_ROOT}/e5_few_shot_summary.csv
  {DATA_ROOT}/e5_few_shot_repeat_details.csv
  {DATA_ROOT}/e5_few_shot_summary.png
"""
import csv
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

import config
from predict_validate import cudem_validation, icesat2_validation, predict_full_scene_tif
from utils import load_region_csvs


PROPORTIONS = [0.00, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]

# Ten repeats are usually enough for reporting mean +/- std in the paper.
# Increase this list if the curves remain unstable.
REPEAT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

METRIC_NAMES = ["R2", "RMSE", "MAE", "Bias", "N"]


def _target_sample(target_df: pd.DataFrame, proportion: float, seed: int) -> pd.DataFrame:
    """Sample target-region training data for one repeat."""
    if proportion <= 0:
        return target_df.iloc[0:0].copy()
    if proportion >= 1.0:
        return target_df.copy()
    n_sample = max(1, int(round(len(target_df) * proportion)))
    return target_df.sample(n=n_sample, random_state=seed)


def train_with_proportion(target_region: str, proportion: float, seed: int = 42):
    """Train one CatBoost model for a target region/proportion/repeat seed."""
    other_regions = [r for r in config.ALL_REGIONS if r != target_region]
    dfs = [load_region_csvs(config.train_dir(r)) for r in other_regions]

    n_target = 0
    if proportion > 0:
        target_df = load_region_csvs(config.train_dir(target_region))
        target_df = _target_sample(target_df, proportion, seed)
        n_target = len(target_df)
        dfs.append(target_df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=config.FEATURE_COLS + [config.TARGET_COL])

    train_idx, valid_idx = train_test_split(df.index, test_size=0.1, random_state=seed)
    train_df = df.loc[train_idx]
    valid_df = df.loc[valid_idx]

    train_pool = Pool(
        train_df[config.FEATURE_COLS],
        label=train_df[config.TARGET_COL],
    )
    valid_pool = Pool(
        valid_df[config.FEATURE_COLS],
        label=valid_df[config.TARGET_COL],
    )

    params = dict(config.CATBOOST_PARAMS)
    params["verbose"] = False
    params["random_seed"] = seed
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    n_train = len(df)

    del train_pool, valid_pool, train_df, valid_df, df, dfs
    gc.collect()
    return model, n_train, n_target


def load_test(region: str) -> pd.DataFrame:
    files = sorted(config.test_dir(region).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No test CSV in {config.test_dir(region)}")
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files],
                   ignore_index=True)
    return df.dropna(subset=config.FEATURE_COLS + [config.TARGET_COL]).reset_index(drop=True)


def _metric_summary(values: list[float]) -> dict:
    arr = np.asarray(values, dtype="float64")
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=1)) if np.isfinite(arr).sum() > 1 else 0.0,
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def summarize_runs(runs: list[dict]) -> dict:
    """Aggregate repeated runs as mean/std for each metric."""
    summary = {
        "n_repeats": len(runs),
        "n_train": _metric_summary([run["n_train"] for run in runs]),
        "n_target": _metric_summary([run["n_target"] for run in runs]),
        "icesat2": {},
        "cudem": {},
    }
    for source in ["icesat2", "cudem"]:
        for metric in METRIC_NAMES:
            summary[source][metric] = _metric_summary([run[source][metric] for run in runs])
    return summary


def _round_metric_dict(metrics: dict) -> dict:
    rounded = {}
    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            rounded[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            rounded[key] = round(float(value), 6)
        else:
            rounded[key] = value
    return rounded


def run_one_repeat(region: str, df_test: pd.DataFrame, proportion: float,
                   repeat_idx: int, seed: int, out_dir: Path) -> dict:
    safe = config.safe_name(region)
    tag = f"{int(proportion * 100):03d}pct"
    repeat_dir = out_dir / f"repeat_{repeat_idx:02d}"
    repeat_dir.mkdir(parents=True, exist_ok=True)

    model, n_train, n_target = train_with_proportion(region, proportion, seed=seed)

    df_test_p = df_test.copy()
    df_test_p["Depth_Pred"] = model.predict(df_test_p[config.FEATURE_COLS]).astype("float32")

    prefix = f"{safe}_few_shot_{tag}_seed{seed}"
    ic_png = repeat_dir / f"{prefix}_icesat2_scatter.png"
    cu_png = repeat_dir / f"{prefix}_cudem_scatter.png"
    ic_m = icesat2_validation(
        df_test_p,
        region,
        ic_png,
        title_suffix=f"Few-shot {proportion * 100:.0f}%, seed {seed}",
    )

    scene_id = config.prediction_scene_id(region)
    tif_path = repeat_dir / f"{prefix}_predicted_depth.tif"
    pred_tif = predict_full_scene_tif(model, region, scene_id, tif_path)
    if pred_tif is None:
        raise RuntimeError(
            f"Full-scene prediction failed for {region} {tag}, seed {seed}; "
            "cannot run CUDEM validation"
        )

    cu_m = cudem_validation(
        pred_tif,
        region,
        cu_png,
        title_suffix=f"Few-shot {proportion * 100:.0f}%, seed {seed}",
    )

    run = {
        "repeat": repeat_idx,
        "seed": seed,
        "proportion": proportion,
        "n_train": int(n_train),
        "n_target": int(n_target),
        "icesat2": _round_metric_dict(ic_m),
        "cudem": _round_metric_dict(cu_m),
        "outputs": {
            "icesat2_scatter": str(ic_png),
            "cudem_scatter": str(cu_png),
            "prediction_tif": str(pred_tif),
        },
    }

    del model, df_test_p
    gc.collect()
    return run


def write_region_detail_csv(region: str, out_dir: Path, region_results: dict) -> None:
    detail_csv = out_dir / "repeat_details.csv"
    rows = []
    for tag, info in region_results.items():
        for run in info["runs"]:
            row = {
                "Region": region,
                "ProportionTag": tag,
                "Proportion": run["proportion"],
                "Repeat": run["repeat"],
                "Seed": run["seed"],
                "N_train": run["n_train"],
                "N_target": run["n_target"],
            }
            for source in ["icesat2", "cudem"]:
                prefix = "ICESat2" if source == "icesat2" else "CUDEM"
                for metric in METRIC_NAMES:
                    row[f"{prefix}_{metric}"] = run[source][metric]
            rows.append(row)

    if rows:
        with open(detail_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[E5] Region repeat details: {detail_csv}")


def main():
    all_results = {}

    for region in config.ALL_REGIONS:
        print(f"\n{'=' * 60}\n[E5 Few-shot repeated] Region = {region}\n{'=' * 60}")
        df_test = load_test(region)
        region_results = {}

        out_dir = config.output_dir(region) / "e5_few_shot"
        out_dir.mkdir(parents=True, exist_ok=True)

        for p in PROPORTIONS:
            tag = f"{int(p * 100)}%"
            print(f"  > Proportion = {p * 100:.0f}% | repeats = {len(REPEAT_SEEDS)}")
            runs = []

            for repeat_idx, seed in enumerate(REPEAT_SEEDS, start=1):
                print(f"    - Repeat {repeat_idx:02d}/{len(REPEAT_SEEDS)} | seed={seed}")
                run = run_one_repeat(region, df_test, p, repeat_idx, seed, out_dir)
                runs.append(run)
                print(
                    "      ICESat-2: "
                    f"RMSE={run['icesat2']['RMSE']:.3f} MAE={run['icesat2']['MAE']:.3f}; "
                    "CUDEM: "
                    f"RMSE={run['cudem']['RMSE']:.3f} MAE={run['cudem']['MAE']:.3f}"
                )

            summary = summarize_runs(runs)
            region_results[tag] = {
                "proportion": p,
                "seeds": list(REPEAT_SEEDS),
                "summary": summary,
                "runs": runs,
            }

            summary_path = out_dir / f"{config.safe_name(region)}_few_shot_{int(p * 100):03d}pct_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(region_results[tag], f, indent=2)

            print(
                f"    Summary ICESat-2 RMSE={summary['icesat2']['RMSE']['mean']:.3f} "
                f"+/- {summary['icesat2']['RMSE']['std']:.3f}; "
                f"CUDEM RMSE={summary['cudem']['RMSE']['mean']:.3f} "
                f"+/- {summary['cudem']['RMSE']['std']:.3f}"
            )

        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(region_results, f, indent=2)
        write_region_detail_csv(region, out_dir, region_results)
        all_results[region] = region_results

    write_global_csvs(all_results)
    plot_summary(all_results)


def write_global_csvs(all_results: dict) -> None:
    summary_csv = config.DATA_ROOT / "e5_few_shot_summary.csv"
    detail_csv = config.DATA_ROOT / "e5_few_shot_repeat_details.csv"

    summary_rows = []
    detail_rows = []
    for region, results in all_results.items():
        for tag, info in results.items():
            summary = info["summary"]
            row = {
                "Region": region,
                "ProportionTag": tag,
                "Proportion": info["proportion"],
                "Repeats": summary["n_repeats"],
                "N_train_mean": round(summary["n_train"]["mean"], 2),
                "N_train_std": round(summary["n_train"]["std"], 2),
                "N_target_mean": round(summary["n_target"]["mean"], 2),
                "N_target_std": round(summary["n_target"]["std"], 2),
            }
            for source in ["icesat2", "cudem"]:
                prefix = "ICESat2" if source == "icesat2" else "CUDEM"
                for metric in METRIC_NAMES:
                    row[f"{prefix}_{metric}_mean"] = round(summary[source][metric]["mean"], 6)
                    row[f"{prefix}_{metric}_std"] = round(summary[source][metric]["std"], 6)
            summary_rows.append(row)

            for run in info["runs"]:
                drow = {
                    "Region": region,
                    "ProportionTag": tag,
                    "Proportion": info["proportion"],
                    "Repeat": run["repeat"],
                    "Seed": run["seed"],
                    "N_train": run["n_train"],
                    "N_target": run["n_target"],
                }
                for source in ["icesat2", "cudem"]:
                    prefix = "ICESat2" if source == "icesat2" else "CUDEM"
                    for metric in METRIC_NAMES:
                        drow[f"{prefix}_{metric}"] = run[source][metric]
                detail_rows.append(drow)

    if summary_rows:
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[E5] Summary CSV: {summary_csv}")

    if detail_rows:
        with open(detail_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"[E5] Repeat detail CSV: {detail_csv}")


def _series(results: dict, source: str, metric: str):
    xs, means, stds = [], [], []
    for _, info in results.items():
        xs.append(info["proportion"] * 100)
        means.append(info["summary"][source][metric]["mean"])
        stds.append(info["summary"][source][metric]["std"])
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(means)[order], np.asarray(stds)[order]


def plot_summary(all_results: dict) -> None:
    colors = {"Florida Bay": "#1f77b4", "Key Largo": "#ff7f0e",
              "Key West": "#2ca02c", "Marathon": "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    panels = [
        ("RMSE", "icesat2", axes[0, 0], "ICESat-2 RMSE (m)"),
        ("RMSE", "cudem", axes[0, 1], "CUDEM RMSE (m)"),
        ("MAE", "icesat2", axes[1, 0], "ICESat-2 MAE (m)"),
        ("MAE", "cudem", axes[1, 1], "CUDEM MAE (m)"),
    ]

    for metric, source, ax, ylabel in panels:
        for region, results in all_results.items():
            xs, means, stds = _series(results, source, metric)
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=3,
                color=colors.get(region, "gray"),
                label=region,
            )
        ax.set_xlabel("Target region samples (%)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel + " vs Target Sample Proportion",
                     fontsize=11, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 5, 10, 20, 40, 60, 80, 100])

    plt.tight_layout()
    out = config.DATA_ROOT / "e5_few_shot_summary.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[E5] Summary plot: {out}")


if __name__ == "__main__":
    main()
