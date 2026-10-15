# Experiment 1 — Main ten-region pooled classification-to-bathymetry experiment

This is the primary controlled comparison. Its executable entry point remains
`../run.py`, and its implementation remains `../workflow.py` so the active
background run is not disrupted.

For each of the seven photon-classification methods, including ATL24 and
Song2024, the workflow concatenates all ten regions and trains five base
regressors plus one validation-constrained ConvexBlend model per spatial outer fold:

1. matches each ATL03 track to the nearest complete Sentinel-2 scene within 45 days;
2. samples eight Sentinel-2 features at method-specific refracted coordinates;
3. applies NOAA MLLW tide normalization and pixel-level 3 x MAD aggregation;
4. assigns complete `track_group_id` values to a five-fold table within each region;
5. uses the Proposed held-out coordinates as a shared spatial anchor and removes
   every method's candidate training/validation samples within 500 m of that
   anchor or its own held-out samples;
6. selects a track-disjoint early-stopping set without restoring rows already
   excluded by the test buffer, then applies the same 500-m rule around validation;
7. trains 2D-CNN, MLP, CatBoost, XGBoost and Random Forest base models;
8. learns non-negative weights summing to one on the validation split only;
9. applies every saved blend to the exact same Proposed-anchor Sentinel-2 pixels,
   samples CUDEM once at those positions, and reports five-fold CUDEM mean ± SD
   as the primary comparison with identical sample counts for every method;
10. retains method-specific ICESat-2 and CUDEM metrics only as supplementary
    diagnostics because their valid sample supports differ by classifier.

This experiment answers whether the photon-classification source changes the
downstream bathymetry result when all ten regions jointly supervise one model.
It is not a leave-one-region-out transfer experiment. CUDEM is the independent
primary test reference; ICESat-2 remains the training target and a supplementary
held-out diagnostic.

Predictions, metrics and reports remain under the main run directory on H:.
Large fitted base-model checkpoints are stored under
`G:/ATL24_ConvexBlend_Checkpoints/ten_region_bathymetry/<run-id>/models/pooled/`.
