from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from experiment_4_temporal_window.workflow import (  # noqa: E402
    add_temporal_columns,
    hemisphere_season,
    load_experiment_config,
    select_stratified_track_groups,
)


class TemporalWindowExperimentTests(unittest.TestCase):
    def test_windows_and_fixed_reference_are_explicit(self):
        cfg = load_experiment_config()
        self.assertEqual(cfg["natural_windows_days"], [7, 15, 30, 45, 60, 90, 180])
        self.assertEqual(cfg["equal_sample_windows_days"], [7, 15, 30, 45, 60, 90, 180])
        self.assertEqual(cfg["primary_reference"], "CUDEM")
        self.assertEqual(cfg["equal_sample_reference_window_days"], 7)
        self.assertEqual(cfg["repeat_seeds"], [42, 43, 44])
        self.assertEqual(cfg["fixed_test_fold"], 0)
        self.assertNotIn("outer_folds", cfg)

    def test_hemisphere_season_is_shifted(self):
        self.assertEqual(hemisphere_season(1, 25.0), "DJF")
        self.assertEqual(hemisphere_season(1, -14.0), "JJA")

    def test_actual_time_delta_controls_nested_windows(self):
        frame = pd.DataFrame({
            "track_time": ["2024-01-01", "2024-01-01"],
            "scene_time": ["2024-01-06", "2024-02-10"],
            "Latitude": [25.0, -14.0],
        })
        result = add_temporal_columns(frame)
        self.assertEqual(result.time_delta_days.tolist(), [5.0, 40.0])
        self.assertEqual(int(result.time_delta_days.le(30).sum()), 1)

    def test_equal_sample_selection_keeps_whole_tracks(self):
        frame = pd.DataFrame({
            "track_group_id": ["a"] * 4 + ["b"] * 5 + ["c"] * 3,
            "region_id": ["r1"] * 4 + ["r1"] * 5 + ["r2"] * 3,
            "track_season": ["DJF"] * 4 + ["MAM"] * 5 + ["DJF"] * 3,
            "Depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4],
        })
        selected, meta = select_stratified_track_groups(frame, 7, 42, [0, 2, 5, 10])
        for group in selected.track_group_id.unique():
            self.assertEqual(len(selected[selected.track_group_id.eq(group)]), len(frame[frame.track_group_id.eq(group)]))
        self.assertEqual(set(selected.track_group_id), set(meta["groups"]))


if __name__ == "__main__":
    unittest.main()
