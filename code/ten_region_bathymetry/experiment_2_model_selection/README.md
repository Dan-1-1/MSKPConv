# Experiment 2 — Regressor model-selection experiment

This experiment uses only the author's Proposed/SKPConv photon-classification,
refraction-correction, and Sentinel-2 matching results from the main workflow.
It does not repeat the five-classifier comparison. For every outer fold, it
concatenates all ten regions and reuses the five learned regressors and constrained
blend already fitted by the main experiment. Only Lyzenga and Stumpf are newly fitted.
All eight regressors are evaluated on the exact CUDEM anchor rows produced by the main
experiment. A second table
reports region-stratified metrics from those same predictions:

- CatBoost (the main paper choice);
- XGBoost;
- Random Forest;
- a standardized two-layer MLP on the eight Sentinel-2 features.
- a lightweight 2-D CNN that receives an 11×11 raw Sentinel-2
  B02/B03/B04/B08 image patch centered on each matched pixel;
- a constrained convex blend of the five learned models above, with non-negative
  weights summing to one and weights fitted only on each outer-fold validation set;
- Lyzenga's linear log-band model;
- Stumpf's blue/green log-ratio model.

The empirical models follow the definitions used in the existing comparison
figure: Lyzenga fits log(B02), log(B03), and log(B04), while Stumpf fits a
linear model to log(1000·B02)/log(1000·B03).

The comparison therefore isolates the regressor choice: the photon classifier,
refraction correction, scene, target, CUDEM reference, and spatial validation protocol
are fixed. The experiment reports eight predictors but performs only ten new fits
(2 empirical models × 5 spatial folds); the other six predictions are generated from
the main experiment's saved checkpoints and blend weights. The CNN is the only image-input model; it does not use the eight
tabular features and instead reads the raw four-band Sentinel-2 window. Fold-level paired
differences are summarized with Wilcoxon signed-rank tests against ConvexBlend,
with Bonferroni-adjusted p-values. Predictions are retained so the paired tests
can be reproduced at the matched test coordinates if required.

```powershell
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_2_model_selection/run.py qc --main-run-id 20261007T003000Z_cudem_anchor
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_2_model_selection/run.py run --main-run-id 20261007T003000Z_cudem_anchor
& 'D:\Python\Anaconda3\envs\env1\python.exe' scripts/ten_region_bathymetry/experiment_2_model_selection/run.py report --main-run-id 20261007T003000Z_cudem_anchor
```

Outputs are stored under:

```text
outputs/ten_region_bathymetry/<main-run-id>/model_selection/regressor_comparison_v1/
├── qc/
├── predictions/
├── metrics/
│   ├── fold_metrics.csv
│   ├── summary.csv
│   ├── convex_blend_weights.csv
│   └── significance_vs_convexblend.csv
└── figures/
```

The two newly fitted empirical-model checkpoints and the copied ConvexBlend weight
metadata are stored under `G:/ATL24_ConvexBlend_Checkpoints/ten_region_bathymetry/`
to avoid filling the H drive.
