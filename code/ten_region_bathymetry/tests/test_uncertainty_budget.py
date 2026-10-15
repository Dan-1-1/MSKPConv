from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE = Path(__file__).parents[1] / "experiment_6_uncertainty_budget" / "workflow.py"
SPEC = importlib.util.spec_from_file_location("uncertainty_workflow", MODULE)
WORKFLOW = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(WORKFLOW)


def test_robust_sigma_ignores_single_outlier() -> None:
    assert WORKFLOW.robust_sigma([0.0, 0.1, -0.1, 0.0, 20.0]) < 0.3


def test_classification_uses_only_requested_model_and_metrics(tmp_path: Path) -> None:
    source = tmp_path / "metrics.csv"
    pd.DataFrame([
        {"seed": seed, "model": model, "OA": oa, "mIoU": 0.8, "Macro-F1": 0.9}
        for seed, oa in [(42, 0.94), (123, 0.95), (2024, 0.93)]
        for model in ["MSKPConv", "baseline"]
    ]).to_csv(source, index=False)
    rows, summary = WORKFLOW.classification_diagnostics(
        {"classification_metrics": str(source), "classification_model": "MSKPConv"})
    assert len(rows) == 3
    assert set(summary.metric) == {"OA", "mIoU", "Macro-F1"}
    assert not summary["atl24_used"].any()
    assert np.isclose(summary.loc[summary.metric.eq("OA"), "mean"].iloc[0], 0.94)


def test_metric_budget_keeps_classification_out_of_metre_variance() -> None:
    oof = pd.DataFrame([{"region_id": "a", "n": 10, "tracks": 2, "rmse_m": 1.0,
                         "mae_m": 0.8, "bias_m": 0.2, "random_u_m": 0.9, "r2": 0.5}])
    sensitivity = pd.DataFrame([{"region_id": "a", "surface_u_m": 0.1, "refraction_u_m": 0.2}])
    tide = pd.DataFrame([{"region_id": "a", "tide_u_m": 0.1}])
    regional, components = WORKFLOW.metric_budget(oof, sensitivity, tide)
    assert regional.loc[0, "classification_unit_conversion"] == "not_performed"
    assert np.isclose(regional.loc[0, "combined_metric_u_m"], 0.9)
    assert "classification" not in set(components.component)

