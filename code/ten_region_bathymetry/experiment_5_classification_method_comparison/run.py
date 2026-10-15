from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from experiment_1_main.workflow import (
    BASE_METHODS,
    METHODS,
    _fit_convex_blend_fold,
    finite_rows,
    load_config,
    model_run_dir,
    regions_from_config,
    run_dir,
    write_json,
)


def comparison_root(cfg: dict, run_id: str) -> Path:
    root = run_dir(cfg, run_id) / "classification_method_comparison"
    root.mkdir(parents=True, exist_ok=True)
    return root


def reused_main_outputs(cfg: dict, run_id: str, command: str) -> dict | None:
    """Return a skip record when the main CV report already contains this comparison.

    The historical experiment-5 entry point retrained seven ConvexBlend models
    on the same splits and the same Proposed-anchor CUDEM pixels that are
    evaluated by ``experiment_1_main.workflow.report``.  Keeping this guard
    makes old pipeline invocations safe while avoiding duplicate training.
    """
    main_root = run_dir(cfg, run_id)
    required = main_root / "report_cudem_anchor_summary.csv"
    fold_metrics = main_root / "report_cudem_anchor_fold_metrics.csv"
    if not (required.exists() and fold_metrics.exists()):
        return None
    root = comparison_root(cfg, run_id)
    result = {
        "run_id": run_id,
        "status": "reused_main_cv_report",
        "command": command,
        "reused": True,
        "source_main_outputs": str(main_root),
        "source_fold_metrics": str(fold_metrics),
        "source_summary": str(required),
        "note": "Skipped redundant seven-method retraining; use experiment_1_main report_cudem_anchor_* outputs.",
    }
    (root / "reused_main_outputs.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def qc(cfg: dict, run_id: str) -> dict:
    root = comparison_root(cfg, run_id)
    main_root = run_dir(cfg, run_id)
    required = [main_root / "evaluation" / "cudem_anchor" / f"fold_{fold}_samples.parquet"
                for fold in range(int(cfg.get("n_folds", 5)))]
    # Some classification products can be unavailable for a single region
    # (currently ATL24 has no Tutuila split).  That is a data-availability
    # condition, not a reason to invalidate the pooled comparison.  Require
    # every method to have at least one split file, while allowing a method's
    # region-level file to be absent when that source has no valid tracks.
    split_root = main_root / "splits"
    available_methods = {
        method: [
            split_root / str(region["id"]) / f"{method}.parquet"
            for region in cfg["regions"]
            if (split_root / str(region["id"]) / f"{method}.parquet").exists()
        ]
        for method in METHODS
    }
    missing_methods = [method for method, paths in available_methods.items() if not paths]
    missing = [str(path) for path in required if not path.exists()]
    missing = [str(path) for path in required if not path.exists()]
    result = {
        "run_id": run_id,
        "status": "PASS" if not missing and not missing_methods else "FAIL",
        "source": str(main_root),
        "required_methods": list(METHODS),
        "missing": missing,
        "missing_methods": missing_methods,
        "available_split_counts": {method: len(paths) for method, paths in available_methods.items()},
        "protocol": "retrain one pooled ConvexBlend per classification/refraction-correction method and fold; evaluate all models on identical Proposed-anchor CUDEM pixels",
    }
    (root / "qc_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if missing or missing_methods:
        details = missing + [f"method_without_any_split:{method}" for method in missing_methods]
        raise RuntimeError("classification-method comparison QC failed: " + ", ".join(details))
    return result


def _summarize(root: Path, fold: pd.DataFrame, run_id: str, main_root: Path) -> dict:
    if "method" not in fold or "fold" not in fold or "support_n" not in fold:
        raise ValueError("anchor fold metrics do not contain method/fold/support_n columns")
    methods = sorted(str(x) for x in fold["method"].dropna().unique())
    expected = sorted(METHODS)
    if methods != expected:
        raise ValueError(f"classification method set mismatch: {methods} != {expected}")
    support = fold.pivot(index="fold", columns="method", values="support_n")
    if support.isna().any().any() or not support.nunique(axis=1).eq(1).all():
        raise RuntimeError("classification methods do not share identical CUDEM test support")

    metric_names = ["rmse", "mae", "bias", "r2", "pearson_r2"]
    summary = fold.groupby("method", as_index=False)[metric_names].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(x).strip("_") for x in summary.columns.to_flat_index()]
    # Keep support explicitly in the table without relying on a fragile index
    # alignment when a method has a missing fold.
    support_rows = fold.groupby("method", as_index=False)["support_n"].agg(
        support_n_min="min", support_n_max="max", support_n_total="sum"
    )
    summary = summary.merge(support_rows, on="method", how="left")
    summary.to_csv(root / "summary_mean_sd.csv", index=False)

    pivot = fold.pivot(index="fold", columns="method", values="rmse").dropna()
    stats: dict[str, object] = {"friedman_rmse": None, "wilcoxon_vs_proposed": []}
    if len(pivot) >= 3 and all(method in pivot.columns for method in METHODS):
        test = friedmanchisquare(*[pivot[method].to_numpy(float) for method in METHODS])
        stats["friedman_rmse"] = {"statistic": float(test.statistic), "p_value": float(test.pvalue), "n_folds": int(len(pivot))}
        for method in BASE_METHODS:
            delta = (pivot[method] - pivot["Proposed"]).dropna()
            if len(delta) >= 2 and np.any(delta.to_numpy(float) != 0):
                result = wilcoxon(delta.to_numpy(float), alternative="two-sided", method="auto")
                stats["wilcoxon_vs_proposed"].append({
                    "method": method, "statistic": float(result.statistic),
                    "p_value": float(result.pvalue), "n_folds": int(len(delta)),
                })
            else:
                stats["wilcoxon_vs_proposed"].append({"method": method, "statistic": None, "p_value": None, "n_folds": int(len(delta))})

    result = {
        "run_id": run_id,
        "status": "completed",
        "classification_methods": expected,
        "regressor": "ConvexBlend",
        "reference": "CUDEM",
        "support": "Proposed_anchor_pixels",
        "identical_support": True,
        "fold_count": int(len(support)),
        "support_n_by_fold": {str(int(index)): int(value) for index, value in support.iloc[:, 0].items()},
        "source_main_outputs": str(main_root),
        "output_root": str(root),
        **stats,
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def run(cfg: dict, run_id: str) -> dict:
    """Retrain one ConvexBlend per classification/refraction method and fold."""
    qc(cfg, run_id)
    root = comparison_root(cfg, run_id)
    main_root = run_dir(cfg, run_id)
    model_root = model_run_dir(cfg, run_id) / "classification_method_comparison" / "models"
    features = list(cfg["features"])
    n_folds = int(cfg.get("n_folds", 5))
    task_dir = root / "tasks"; task_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = root / "metrics"; metrics_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    metrics_path = metrics_dir / "fold_metrics.csv"
    for method in METHODS:
        parts = []
        for region in regions_from_config(cfg):
            path = main_root / "splits" / region.id / f"{method}.parquet"
            if path.exists():
                data = pd.read_parquet(path)
                if not data.empty:
                    parts.append(data)
        if not parts:
            continue
        data = pd.concat(parts, ignore_index=True)
        for fold in range(n_folds):
            task_path = task_dir / f"{method}__fold{fold}.json"
            if task_path.exists():
                rows.extend(json.loads(task_path.read_text(encoding="utf-8")))
                continue
            split_col = f"split_fold_{fold}"
            if split_col not in data.columns:
                continue
            train = finite_rows(data[data[split_col].eq("train") & (data.fold != fold)], features + ["Depth"])
            validation = finite_rows(data[data[split_col].eq("validation") & (data.fold != fold)], features + ["Depth"])
            anchor_path = main_root / "evaluation" / "cudem_anchor" / f"fold_{fold}_samples.parquet"
            test = finite_rows(pd.read_parquet(anchor_path), features + ["Depth", "CUDEM_Depth"])
            if len(train) < int(cfg.get("min_train_rows", 500)) or len(validation) < int(cfg.get("min_validation_rows", 100)) or test.empty:
                task_path.write_text(json.dumps([], ensure_ascii=False), encoding="utf-8")
                continue
            test = test.copy(); test["method"] = method
            destination = model_root / method / f"fold_{fold}"
            prediction, weights, info = _fit_convex_blend_fold(
                cfg, train, validation, test, destination, fold,
                int(cfg.get("seed", 42)) + fold, True,
            )
            blend_info = {
                "type": "classification_method_comparison_convex_blend",
                "method": method, "fold": fold, "base_regressors": info["base_regressors"],
                "weights": weights.tolist(), "train_rows": len(train),
                "validation_rows": len(validation), "test_rows": len(test),
                **{k: v for k, v in info.items() if k not in {"base_regressors", "weights"}},
            }
            write_json(destination / "ConvexBlend" / f"fold_{fold}.json", blend_info)
            out = test[["region_id", "method", "fold", "track_group_id", "scene_id",
                        "img_row", "img_col", "Latitude", "Longitude", "Depth", "CUDEM_Depth"]].copy()
            out["Prediction"] = prediction
            pred_path = root / "predictions" / method / f"fold_{fold}.parquet"
            pred_path.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(pred_path, index=False)
            from experiment_1_main.workflow import metrics
            overall = metrics(out.CUDEM_Depth.to_numpy(float), prediction)
            task_rows = [{"scope": "pooled", "method": method, "fold": fold,
                          "regressor": "ConvexBlend", "status": "OK",
                          "reference": "CUDEM", "support": "Proposed_anchor_pixels",
                          "support_n": len(out), "train_rows": len(train),
                          "validation_rows": len(validation), **overall}]
            for region_id, group in out.groupby("region_id"):
                region_metrics = metrics(group.CUDEM_Depth.to_numpy(float), group.Prediction.to_numpy(float))
                task_rows.append({"scope": "region", "region_id": region_id,
                                  "method": method, "fold": fold, "regressor": "ConvexBlend",
                                  "status": "OK", "reference": "CUDEM",
                                  "support": "Proposed_anchor_pixels", "support_n": len(group),
                                  **region_metrics})
            task_path.write_text(json.dumps(task_rows, ensure_ascii=False), encoding="utf-8")
            rows.extend(task_rows)
            pd.DataFrame(rows).to_csv(metrics_path, index=False)
            print(f"classification comparison: {method}/fold{fold} train={len(train)} val={len(validation)} test={len(test)}", flush=True)
    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(metrics_path, index=False)
    pooled = fold_metrics[fold_metrics.scope.eq("pooled")].copy()
    return _summarize(root, pooled, run_id, main_root)


def report(cfg: dict, run_id: str) -> dict:
    root = comparison_root(cfg, run_id)
    path = root / "metrics" / "fold_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return _summarize(root, pd.read_csv(path).query("scope == 'pooled'"), run_id, run_dir(cfg, run_id))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare classification/refraction-correction methods on identical CUDEM support")
    parser.add_argument("command", choices=["qc", "run", "report"])
    parser.add_argument("--main-run-id", required=True)
    parser.add_argument("--main-config", type=Path, default=PARENT / "experiment_1_main" / "config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.main_config)
    reused = reused_main_outputs(cfg, args.main_run_id, args.command)
    if reused is not None:
        print(json.dumps(reused, ensure_ascii=False))
        return 0
    if args.command == "qc":
        result = qc(cfg, args.main_run_id)
    elif args.command == "run":
        result = run(cfg, args.main_run_id)
    else:
        result = report(cfg, args.main_run_id)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
