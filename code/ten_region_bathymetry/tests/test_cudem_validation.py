from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
from experiment_1_main.workflow import assign_validation_groups, depth_domain_mask, load_config  # noqa: E402


class CudemValidationTests(unittest.TestCase):
    def test_full_scene_domain_keeps_deep_bathymetry_and_excludes_land(self):
        values = pd.Series([-483.0, -0.1, 0.0, 7.5, 20.0, 20.1, 450.0])
        keep = depth_domain_mask(values, {"evaluation_min_m": 0.0, "evaluation_max_m": None})
        self.assertEqual(values[keep].tolist(), [0.0, 7.5, 20.0, 20.1, 450.0])

    def test_validation_assignment_does_not_restore_spatially_excluded_rows(self):
        frame = pd.DataFrame(
            {
                "track_group_id": ["selected", "selected", "other"],
                "split_fold_0": ["train", "excluded_spatial_buffer", "train"],
            }
        )

        result = assign_validation_groups(frame, "split_fold_0", ["selected"])

        self.assertEqual(
            result["split_fold_0"].tolist(),
            ["validation", "excluded_spatial_buffer", "train"],
        )

    def test_primary_evaluation_uses_proposed_anchor_and_cudem(self):
        cfg = load_config(PACKAGE_ROOT / "experiment_1_main" / "config.yaml")
        self.assertEqual(cfg["shared_test_anchor_method"], "Proposed")
        self.assertEqual(cfg["primary_evaluation_reference"], "CUDEM")


if __name__ == "__main__":
    unittest.main()
