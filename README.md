# MSKPConv

Reproducibility release for the manuscript **MSKPConv: ICESat-2 Photon
Classification for Sentinel-2 Shallow-Water DEM Reconstruction**.

This revision replaces the former single-model inversion package with the final
ten-region **ConvexBlend** workflow. The inversion model is a non-negative,
sum-to-one blend of five fitted regressors:

1. Lightweight 2-D CNN
2. multilayer perceptron (MLP)
3. CatBoost
4. XGBoost
5. random forest

The repository contains model code, inference entry points, trained
classification weights, ConvexBlend metadata and small base-model weights, and
the 220 labeled ATL03 classification CSV files used in the experiments, 17
preserved historical source/alias CSV files, four labeled `ATL24_Test` files,
and the exact split indices used by the reported experiments. The complete final
ConvexBlend bundle, including the 500 MiB random-forest checkpoint, is attached
to GitHub Release `v2.0.0` because that file exceeds GitHub's 100 MiB regular
file limit.

## Reviewer requirement 18

| Required item | Location |
| --- | --- |
| Trained MSKPConv and classification-baseline weights | `models/classification/` |
| Complete five-model ConvexBlend weights | GitHub Release `v2.0.0`; manifest in `models/WEIGHTS_MANIFEST.json` |
| Photon classification inference | `code/mskpconv/evaluate_mskpconv.py` |
| 220 experimental ATL03 CSV files and 17 preserved historical aliases | `data/Train/`, `data/Val/`, `data/Test/`; exact 220-file membership in `splits/photon_classification/files.csv` |
| Four labeled ATL24 external-test CSV files | `data/ATL24_Test/` |
| Five-model ConvexBlend inference | `code/ten_region_bathymetry/predict_convexblend.py` |
| Exact classification file split | `splits/photon_classification/files.csv` |
| Exact inversion row/fold assignments | `splits/convexblend/row_splits.parquet` |
| Human-readable track/fold index | `splits/convexblend/track_folds.csv` |
| Five outer-fold blend weights | `models/convexblend/cross_validation_weights/` |
| Exact experiment configuration | `code/ten_region_bathymetry/experiment_1_main/config.yaml` |

`splits/convexblend/row_splits.parquet` stores every experimental row's source
index, region, classification method, track group, Sentinel-2 scene and pixel,
geographic coordinates, outer fold, and its train/validation/test/excluded
status for each of the five folds. This is the authoritative split artifact;
summary counts alone are not used as a substitute for indices.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/download_convexblend_weights.py
python scripts/verify_release.py
```

For CUDA, install the PyTorch wheel matching the local driver instead of the
CPU/default wheel resolved by the last command.

## ConvexBlend inference

The table input must contain the eight columns listed below. The CNN input is a
NumPy array with shape `[N, 4, 11, 11]` in B02/B03/B04/B08 order and with rows
aligned to the table.

```bash
python code/ten_region_bathymetry/predict_convexblend.py \
  --input features.parquet \
  --patches sentinel2_patches.npy \
  --output predictions.parquet
```

Required tabular features are `B02`, `B03`, `B04`, `B08`, `NDWI`,
`Blue_Green_LogRatio`, `Blue_Red_LogRatio`, and `Green_Red_LogRatio`.
The command writes all five base predictions and the final convex combination,
so the blend can be audited row by row.

## Photon classification inference

```bash
python code/mskpconv/evaluate_mskpconv.py \
  --checkpoint models/classification/mskpconv/mskpconv_best.pth \
  --test-dir data/Test
```

The exact train/validation/test file allocation is independent of the local
data path and is recorded in `splits/photon_classification/files.csv` with file
sizes, row counts, and SHA-256 hashes.

## Full experiment

The exact research workflow and configuration are under
`code/ten_region_bathymetry/`. The archived configuration intentionally records
the original drive paths and run settings. For a new machine, copy the YAML and
change only `workspace_root`, `data_root`, `output_root`, and
`model_output_root`; do not alter the seed, folds, spatial buffer, features, or
model hyperparameters when reproducing the paper.

```bash
python code/ten_region_bathymetry/experiment_1_main/run.py qc --run-id RUN_ID
python code/ten_region_bathymetry/experiment_1_main/run.py prepare --run-id RUN_ID
python code/ten_region_bathymetry/experiment_1_main/run.py split --run-id RUN_ID
python code/ten_region_bathymetry/experiment_1_main/run.py train --run-id RUN_ID
python code/ten_region_bathymetry/experiment_1_main/run.py report --run-id RUN_ID
```

See `docs/REPRODUCIBILITY.md` for split provenance, weight provenance, and the
limits of the public package.
