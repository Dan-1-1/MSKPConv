# Trained weights

`classification/` contains the released MSKPConv, PointNet++, PointNeXt,
KPConv, and DGCNN checkpoints. The `mskpconv_best_baseline_package.pt` file is
the MSKPConv-format checkpoint used by the unified baseline evaluator; the
standalone model uses `classification/mskpconv/mskpconv_best.pth`.

`convexblend/final/Proposed/` contains the final ten-region deployment model.
The repository directly stores the four base checkpoints below GitHub's
100 MiB regular-file limit and `ConvexBlend/final.json`. The complete directory,
including `RandomForest/final.joblib`, is in Release `v2.0.0`.

`convexblend/cross_validation_weights/` contains the five independently fitted
outer-fold blend metadata files. These are the source of the averaged final
weights and preserve the exact mapping between base-regressor order and weight.

Every checksum is recorded in `WEIGHTS_MANIFEST.json`.
