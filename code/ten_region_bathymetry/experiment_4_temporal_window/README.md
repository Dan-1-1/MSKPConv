# Experiment 4 — Temporal-window robustness and seasonal bias

This experiment answers the reviewer concern that a wide ICESat-2/Sentinel-2
matching window may appear superior only because it supplies more training
samples or happens to align favorably with the fixed test season.

## Design

The workflow matches every Proposed-classification track once using the maximum
180-day window and records its actual absolute time difference. The 7, 15, 30,
45, 60, 90 and 180-day datasets are nested filters of this single underlying
population, so differences cannot arise from separate source inventories.

Two controlled branches are fitted with the same ConvexBlend model:

1. `natural`: retain every eligible complete track in each window. This measures
   the operational trade-off between sample quantity and temporal consistency.
2. `equal_sample`: use the 30-day training capacity for every wider window and
   select complete track groups with repeated region × depth × hemisphere-season
   stratification. This isolates the temporal-window effect from sample count.

Both branches reuse the main experiment's fixed Proposed outer-fold test rows,
500 m exclusion zones, Sentinel-2 features and CUDEM reference. Test coordinates
never change across windows. Seasonal-bias output includes per-region and
per-region-season metrics, actual time-difference distributions and the fraction
of training samples whose ICESat-2 and Sentinel-2 dates lie in different local
meteorological seasons.

## Commands

```powershell
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_4_temporal_window/run.py qc --main-run-id <RUN_ID>
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_4_temporal_window/run.py prepare --main-run-id <RUN_ID>
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_4_temporal_window/run.py run --main-run-id <RUN_ID>
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_4_temporal_window/run.py report --main-run-id <RUN_ID>
```

Use `--max-tasks 1` for a smoke test. `--windows`, `--folds` and `--seeds`
accept comma-separated subsets. Every task is resumable.

## Output layout

```text
outputs/ten_region_bathymetry/<RUN_ID>/experiment_4_temporal_window/temporal_window_robustness_v1/
├── qc/summary.json
├── prepared/
│   ├── master/<region>/Proposed.parquet
│   ├── noaa_tide_cache.json
│   └── summary.json
├── metrics/
│   ├── tasks/*.json
│   ├── task_metrics.csv
│   ├── summary_mean_sd.csv
│   ├── paired_window_tests.csv
│   ├── sample_and_season_diagnostics.csv
│   └── sample_distribution_by_region_season.csv
└── report_summary.json
```

