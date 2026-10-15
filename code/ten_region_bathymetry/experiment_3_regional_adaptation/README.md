# Experiment 3 — Ten-region regional-adaptation experiment

This experiment is separate from the main seven-method comparison. For each
target region, it trains on the other nine source regions and adds 0%, 5%, 10%,
20%, 40%, 60%, 80% or 100% of the eligible target-region training pool.

The target pool and test set come from the main experiment's strict split:

- the outer-fold test track groups remain fixed for every proportion and seed;
- CUDEM is sampled once on each fixed target-fold test support and is the
  primary reference for every proportion and repeat seed;
- the corresponding Proposed ICESat-2 depths are retained only as
  supplementary diagnostics and adaptation-training targets;
- target adaptation samples are selected as complete `track_group_id` values;
- the main split has already removed samples within 500 m of the target test set;
- group selection minimizes both sample-count error and depth-bin imbalance;
- the source validation set is a fixed, spatially blocked fold and never uses
  target test samples;
- each proportion is repeated with ten seeds and summarized as mean and sample
  standard deviation.

The nominal percentage is a sampling target. Because tracks are indivisible,
the workflow records the actual sample percentage and number of selected tracks.
The default run uses all five outer spatial folds, eight target proportions,
and ten repeat seeds (10 regions × 5 folds × 8 proportions × 10 seeds = 4,000
ConvexBlend tasks for the default Proposed method). Each task trains 2D-CNN,
MLP, CatBoost, XGBoost and Random Forest, then estimates non-negative weights
summing to one on the source-region validation split. Target test tracks never
enter base-model fitting or weight estimation.

The default configuration runs only the Proposed/SKPConv chain. This isolates
regional adaptation from the five-method classification comparison and keeps the
experiment computationally tractable. Additional methods can be listed in
`config.yaml` without changing the code.

## Commands

```powershell
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_3_regional_adaptation/run.py qc --main-run-id 20261007T003000Z_cudem_anchor
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_3_regional_adaptation/run.py run --main-run-id 20261007T003000Z_cudem_anchor
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_3_regional_adaptation/run.py report --main-run-id 20261007T003000Z_cudem_anchor
```

Outputs are written below:

```text
outputs/ten_region_bathymetry/<main-run-id>/regional_adaptation/track_adaptation_convex_v1/
├── qc/
├── assignments/
├── models/
├── predictions/
├── metrics/
├── figures/
├── config_snapshot.json
└── run_status.json
```

## Spatially independent publication figures

The redesigned CUDEM evaluation uses one fixed validation mask per region. The
mask excludes a 500 m buffer around every candidate target-adaptation track and
is unchanged across adaptation shares. Generate the main regional RMSE figure,
supplementary heatmaps, exact sample-count tables, and captions with:

```powershell
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_3_regional_adaptation/plot_spatially_independent_results.py
```

The plotting script does not treat an error measured on excluded ICESat-2 track
support as a CUDEM lower bound. A reference-error band is accepted only through
`--reference-error-csv` when the file explicitly identifies an independent
estimate on the exact fixed 500 m evaluation support. Without such an estimate,
the figure shows the 100% target-data result as an empirical benchmark rather
than a theoretical lower bound.
