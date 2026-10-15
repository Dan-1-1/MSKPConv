from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
from scene_evaluation.workflow import (  # noqa: E402
    _pooled_metrics,
    _sufficient_statistics,
    experiment_root,
    load_scene_config,
    raster_root,
)


class SceneEvaluationTests(unittest.TestCase):
    def test_all_ten_regions_and_final_model_are_configured(self):
        cfg = load_scene_config()
        self.assertEqual(
            cfg["regions"],
            [
                "florida_bay",
                "key_largo",
                "key_west",
                "marathon",
                "anegada",
                "tutuila",
                "cocos_guam",
                "kahului",
                "saipan",
                "kaneohe",
            ],
        )
        self.assertEqual(cfg["methods"], ["Proposed"])
        self.assertEqual(cfg["support"], "proposed_convexblend_valid_pixels")
        self.assertEqual(cfg["depth_domain"], "full_scene_cudem_bathymetry")
        self.assertEqual(cfg["model_variant"], "final_full")
        self.assertFalse(cfg["save_uncertainty"])
        self.assertNotIn("max_depth_m", cfg)

    def test_pooled_metrics_match_direct_calculation(self):
        y1 = np.array([1.0, 2.0, 3.0])
        p1 = np.array([1.5, 1.5, 3.0])
        y2 = np.array([4.0, 5.0])
        p2 = np.array([3.0, 6.0])
        rows = []
        for y, pred in ((y1, p1), (y2, p2)):
            rows.append({"n": len(y), **_sufficient_statistics(y, pred)})
        pooled = _pooled_metrics(pd.DataFrame(rows))
        y = np.concatenate([y1, y2]); pred = np.concatenate([p1, p2])
        self.assertAlmostEqual(pooled["rmse"], float(np.sqrt(np.mean((pred - y) ** 2))))
        self.assertAlmostEqual(pooled["mae"], float(np.mean(np.abs(pred - y))))
        self.assertAlmostEqual(pooled["bias"], float(np.mean(pred - y)))

    def test_output_roots_are_separated(self):
        with tempfile.TemporaryDirectory() as metrics_tmp, tempfile.TemporaryDirectory() as raster_tmp:
            main = {"output_root": metrics_tmp}
            scene = {"experiment_id": "scene", "raster_output_root": raster_tmp}
            self.assertEqual(experiment_root(main, scene, "run"), Path(metrics_tmp) / "run" / "scene_evaluation" / "scene")
            self.assertEqual(raster_root(main, scene, "run"), Path(raster_tmp) / "run" / "scene")


if __name__ == "__main__":
    unittest.main()
