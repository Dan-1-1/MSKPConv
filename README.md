# MSKPconv

Code and trained model weights for the manuscript:

**MSKPConv: ICESat-2 Photon Classification for Sentinel-2 Shallow-Water DEM Reconstruction**

This GitHub release contains the core source code and trained weights used for manuscript review. Large training, inversion, DEM, and Sentinel-2 data files are not included in this repository because of their size. Data can be shared through a research data repository or by request.

## Repository Structure

```text
code/
  mskpconv/                  MSKPConv model, preprocessing, training, evaluation, and prediction
  baseline_classification/   PointNet++, PointNeXt, KPConv, DGCNN, and ablation experiments
  photon_correction/         ICESat-2 refraction correction and DEM-based point validation
  image_matching/            ICESat-2/Sentinel-2 matching utilities
  catboost_inversion/        CatBoost bathymetry inversion and validation experiments
models/
  mskpconv/                  Trained MSKPConv checkpoint
  baseline_classification/   Baseline and ablation model checkpoints
  catboost/                  Region-specific CatBoost inversion models
requirements.txt             Python package requirements
```

## Notes

- Model weights are included directly in the repository because each file is below GitHub's 100 MB single-file limit.
- The original local reviewer package also contains data and result products, but those directories are intentionally excluded from this GitHub repository.
- Paths in the scripts are organized relative to the release package where possible; update local data paths before rerunning full experiments.
