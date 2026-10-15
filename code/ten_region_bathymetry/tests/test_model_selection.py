from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
from experiment_2_model_selection.workflow import _finite_rows, _fit_convex_weights, load_selection_config, selection_model_root, selection_root  # noqa: E402


class ModelSelectionTests(unittest.TestCase):
    def test_regressors_and_convex_blend_are_configured(self):
        cfg = load_selection_config()
        self.assertEqual(cfg["methods"], ["Proposed"])
        self.assertEqual(cfg["training_mode"], "pooled")
        self.assertEqual(cfg["primary_reference"], "CUDEM")
        self.assertEqual(cfg["test_support"], "fixed_proposed_fold_cudem")
        self.assertEqual(cfg["regressors"], ["CatBoost", "XGBoost", "RandomForest", "MLP", "LightweightCNN", "ConvexBlend", "Lyzenga", "StumpfLogRatio"])
        self.assertEqual(len(cfg["folds"]), 5)
        self.assertEqual(cfg["lightweight_cnn"]["image_patch_size"], 11)
        self.assertEqual(cfg["convex_blend"]["base_regressors"], ["LightweightCNN", "MLP", "CatBoost", "XGBoost", "RandomForest"])

    def test_finite_rows_removes_nan_and_infinity_on_shared_support(self):
        frame = pd.DataFrame(
            {
                "B02": [0.1, np.inf, 0.3, 0.4],
                "Ratio": [1.0, 2.0, -np.inf, 4.0],
                "Depth": [2.0, 3.0, 4.0, np.nan],
                "row_id": ["keep", "inf_band", "inf_ratio", "nan_depth"],
            }
        )

        clean = _finite_rows(frame, ["B02", "Ratio", "Depth"])

        self.assertEqual(clean["row_id"].tolist(), ["keep"])
        self.assertTrue(np.isfinite(clean[["B02", "Ratio", "Depth"]].to_numpy()).all())

    def test_convex_weights_are_nonnegative_and_sum_to_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        predictions = np.column_stack([y, y + 1.0, np.full(4, 10.0)])
        weights = _fit_convex_weights(y, predictions)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
        self.assertGreater(float(weights[0]), 0.99)

    def test_selection_root_is_created(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = selection_root(
                {"output_root": tmp},
                "run",
                {"experiment_id": "selection"},
            )
            self.assertTrue(root.is_dir())

    def test_selection_models_use_configured_large_output_root(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as model_tmp:
            root = selection_model_root(
                {"output_root": tmp, "model_output_root": model_tmp},
                "run",
                {"experiment_id": "selection"},
            )
            self.assertEqual(root, Path(model_tmp) / "run" / "model_selection" / "selection")


if __name__ == "__main__": unittest.main()
