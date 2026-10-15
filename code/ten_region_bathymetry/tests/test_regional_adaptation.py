from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from experiment_3_regional_adaptation.workflow import (  # noqa: E402
    load_adaptation_config, select_depth_balanced_groups, select_nested_track_groups,
)


class RegionalAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "track_group_id": ["a"] * 3 + ["b"] * 4 + ["c"] * 5 + ["d"] * 2,
            "Depth": [1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 16, 4, 5,
                      ],
        })

    def test_selection_is_whole_track_and_reproducible(self):
        a, meta_a = select_depth_balanced_groups(self.df, 0.5, 42, [0, 2, 5, 10, 15, 20], 0.5)
        b, meta_b = select_depth_balanced_groups(self.df, 0.5, 42, [0, 2, 5, 10, 15, 20], 0.5)
        self.assertEqual(meta_a["groups"], meta_b["groups"])
        self.assertEqual(set(a.track_group_id), set(meta_a["groups"]))
        self.assertTrue(all(len(a[a.track_group_id == g]) == len(self.df[self.df.track_group_id == g]) for g in meta_a["groups"]))

    def test_zero_and_full_proportions(self):
        zero, zmeta = select_depth_balanced_groups(self.df, 0.0, 1, [0, 5, 10], 0.5)
        full, fmeta = select_depth_balanced_groups(self.df, 1.0, 1, [0, 5, 10], 0.5)
        self.assertEqual(len(zero), 0); self.assertEqual(zmeta["groups"], [])
        self.assertEqual(len(full), len(self.df)); self.assertEqual(set(full.track_group_id), set(self.df.track_group_id))

    def test_cudem_is_primary_full_scene_reference(self):
        cfg = load_adaptation_config()
        self.assertEqual(cfg["primary_reference"], "CUDEM")
        self.assertEqual(cfg["test_support"], "full_representative_scene_underwater_cudem")
        self.assertEqual(cfg["adaptation_strategy"], "leave_one_region_out_full_model_retraining")
        self.assertEqual(cfg["selection_seeds"], [42])
        self.assertEqual(len(cfg["target_regions"]), 10)
        self.assertEqual(cfg["proportions"], [0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
        self.assertEqual(len(cfg["target_regions"]) * len(cfg["proportions"]) * len(cfg["selection_seeds"]), 60)
        self.assertNotIn("outer_folds", cfg)
        self.assertNotIn("fixed_fold", cfg)

    def test_nested_selection_is_monotonic(self):
        frame = self.df.copy()
        frame["x"] = range(len(frame)); frame["y"] = range(len(frame))
        frame["scene_id"] = "S2_Mosaic_202401_Cropped_CloudMasked"
        result = select_nested_track_groups(frame, [0, .1, .5, 1], 42, [0, 2, 5, 10, 20], 1000)
        previous = set()
        for proportion in [0, .1, .5, 1]:
            current = set(result[proportion][1]["groups"])
            self.assertTrue(previous.issubset(current)); previous = current
        self.assertEqual(previous, set(frame.track_group_id))


if __name__ == "__main__":
    unittest.main()
