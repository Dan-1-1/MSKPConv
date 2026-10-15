# Ten-region bathymetry workflow

This directory implements the configuration-driven experiment in
`十区域多分类结果水深反演与对比实验方案`. It never writes into the source CSV or
Sentinel-2 directories. Generated artefacts go to `outputs/ten_region_bathymetry`.

## Directory layout

```text
ten_region_bathymetry/
├── experiment_1_main/
│   ├── run.py                     Main experiment CLI
│   ├── workflow.py                Pooled ten-region training and evaluation
│   ├── config.yaml                Shared data and ConvexBlend configuration
│   └── README.md                  Experiment 1 protocol and outputs
├── experiment_2_model_selection/
│   ├── run.py                     Eight-regressor comparison CLI
│   ├── workflow.py                Model fitting, inference and statistics
│   ├── config.yaml                Regressor settings and folds
│   └── README.md                  Experiment 2 protocol and outputs
├── experiment_3_regional_adaptation/
│   ├── run.py                     Regional-adaptation CLI
│   ├── workflow.py                Leave-one-region-out few-shot pipeline
│   ├── config.yaml                Proportions, folds, repeats and caps
│   └── README.md                  Experiment 3 protocol and outputs
├── experiment_4_temporal_window/
│   ├── run.py                     Experiment 4 CLI
│   ├── workflow.py                Nested-window, equal-sample and seasonal analysis
│   ├── config.yaml                Windows, repeats, folds and sampling protocol
│   └── README.md                  Reviewer-facing design and output contract
├── scene_evaluation/              Auxiliary full-scene CUDEM evaluation
├── tests/                         Unit and protocol tests
├── run_convex_pipeline.py         Ordered Experiment 1→4 runner
├── run.py / workflow.py           Experiment 1 compatibility entry points
├── model_selection/               Experiment 2 compatibility entry points
└── regional_adaptation/           Experiment 3 compatibility entry points
```

The three experiments answer different questions. The main experiment now
concatenates all ten regions and trains one pooled model per each of seven
classification methods (including ATL24 and Song2024) and spatial fold, while
using identical Proposed-anchor pixels and independent CUDEM depths as its
primary test support. Region-stratified method-specific ICESat-2 metrics are
retained only as supplementary diagnostics.
The regional-adaptation experiment measures zero-shot transfer and the benefit
of adding target-region track groups. The latter consumes the former's prepared
and split Parquet files but has its own models, metrics, logs and figures. The
model-selection experiment holds the classifier and spatial split fixed while
comparing CatBoost, XGBoost, Random Forest, MLP, 2D-CNN and ConvexBlend regressors.
Experiment 4 runs after regional adaptation. It combines natural-sample temporal
windows, repeated track-level equal-sample controls, and per-region seasonal-bias
tests on the same spatially isolated CUDEM support.

Use the environment containing the geospatial, PyTorch and tree-model stack:

```powershell
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_1_main/run.py qc
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_1_main/run.py prepare
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_1_main/run.py split
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_1_main/run.py train
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_1_main/run.py report
```

The default run id is UTC-based and can be fixed with `--run-id` for exact
reproduction. `qc` is offline. `prepare` requires NOAA CO-OPS water-level
observations; a request failure marks the affected scene as skipped and never
substitutes a zero tide. The strict split is shared by all seven classification
methods and uses a 500-m exclusion around Proposed anchor test coordinates.

The first execution should be staged as `qc`, then `prepare`, `split`, and a
small `train --max-tasks 1` smoke test before launching the full 35-fold run.

When one region is replaced but the other nine source datasets are unchanged,
the ordered pipeline can seed their immutable prepared Parquet files from a
completed run and rematch only the replacement region:

```powershell
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/run_convex_pipeline.py `
  --run-id 20261008T000000Z_tutuila `
  --prepared-source-run-id 20261007T120000Z_full_depth `
  --rebuild-region tutuila
```

The new run records this provenance in `prepared/reuse_lineage.json`; spatial
fold assignment, all regressors, model selection, representative-scene CUDEM
evaluation, and regional adaptation are then recomputed from the revised
ten-region sample set.
