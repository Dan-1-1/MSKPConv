# MSKPConv

Reproducible implementation of **MSKPConv: ICESat-2 Photon Classification for Sentinel-2 Shallow-Water DEM Reconstruction**. The repository contains the current model, the 24-dimensional handcrafted feature extractor, training/evaluation scripts, baseline implementations, and the previously uploaded datasets and checkpoints.

## Repository layout

```text
code/
  mskpconv/
    config.py                       portable paths and experiment settings
    preprocess_atl03_features.py    24-D feature extraction and diagnostics
    convert_hdf5.py                 optional HDF5 cache generation
    dataset.py                      chunked point-cloud datasets and voting
    mskpconv_model.py               MSKPConv encoder-decoder network
    losses.py                       classification and Lovász losses
    train_mskpconv.py               training, validation, and test evaluation
    evaluate_mskpconv.py            standalone checkpoint evaluation
    predict_validation_regions.py   prediction on unlabeled regional files
    visualize_photon_predictions.py plots and per-file prediction exports
  baseline_classification/          PointNet++, PointNeXt, KPConv, DGCNN, and ablations
  photon_correction/                ICESat-2 refraction and DEM validation utilities
  image_matching/                   ICESat-2/Sentinel-2 matching utilities
  catboost_inversion/               CatBoost bathymetry inversion utilities
data/
  Train/ Val/ Test/                 labeled ATL03 CSV files
models/                              released checkpoints
requirements.txt                    Python dependencies
```

Generated logs, predictions, caches, and figures are written under `results/` and are ignored by Git. Existing `data/` and `models/` files are not changed by the code update.

## 24-dimensional handcrafted features

The extractor returns features in this fixed order:

| Dimensions | Feature group | Default settings |
|---|---|---|
| 1–4 | Elevation difference | K = 5, 10, 15, 25 |
| 5–8 | LOFE / weighted neighborhood distance | K = 5, 10, 15, 25 |
| 9–12 | Density | radii = 1, 2, 4, 5 |
| 13–16 | Local vertical standard deviation | K = 5, 10, 15, 25 |
| 17–20 | Vertical asymmetry | radii = 1, 2, 4, 5; thresholds = 0.05, 0.10, 0.15, 0.20 |
| 21–24 | Horizontal asymmetry | radii = 1, 2, 4, 5; thresholds = 0.5, 1.0, 1.5, 2.0 |

The KNN and density groups use the anisotropic scale pairs `(s_x, s_z)` = `(1/5, 1)`, `(1/10, 2)`, `(1/15, 2)`, and `(1/20, 4)`. File-level group normalization is applied after feature extraction. Depth-dependent pull-factor correction and neighbor diagnostics are retained in the current implementation.

## Reproduction workflow

Run commands from the repository root after installing the dependencies:

```bash
pip install -r requirements.txt
python code/mskpconv/convert_hdf5.py
python code/mskpconv/train_mskpconv.py
python code/mskpconv/evaluate_mskpconv.py
```

`convert_hdf5.py` is optional when CSV preprocessing is preferred. The cache is configuration-dependent; regenerate it whenever the feature settings change. The default evaluation reads the independent `data/Test` split, which contains the 20 test tracks used for the reported ATL03 evaluation.

## Reproducible overrides

The default paths are repository-relative. For a custom data/output location, set:

```bash
export MSKPCONV_DATA_ROOT=/path/to/data
export MSKPCONV_OUTPUT_ROOT=/path/to/results
export MSKPCONV_MODEL_ROOT=/path/to/models
export MSKPCONV_DEVICE=cuda:0
```

For a sensitivity run, provide a JSON file with `knn_configs` and/or `density_configs` and set `FEATURE_CONFIG_PATH`. `EXPERIMENT_ROOT`, `EXPERIMENT_SEED`, `SENSITIVITY_EPOCHS`, `SENSITIVITY_EARLY_STOP_PATIENCE`, `USE_HDF5_CACHE`, and `AUTO_PREDICT_REGIONS` are also supported by `config.py`.

## Evaluation outputs

Training writes the best checkpoint, epoch history, confusion matrix, pooled metrics, and per-file metrics under the configured output directory. The per-file export preserves the file identity so that pooled metrics, regional summaries, and paired track-level comparisons can be reproduced without treating photons from different tracks as independent experimental units.

## License and citation

Please cite the accompanying manuscript when using this implementation. Check the repository history and manuscript for the applicable license and citation details.
