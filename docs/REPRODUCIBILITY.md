# Reproducibility record

## Final inversion model

The release corresponds to local experiment run
`20261011T000000Z_selected_r2`. The final deployment model was fitted on
132,916 finite samples from ten regions. Its five ConvexBlend weights are the
mean of weights learned independently on the five outer-fold validation sets,
then clipped to be non-negative and normalized to sum to one. The final base
learners were refitted on all finite ten-region samples.

The five base regressors, in stored order, are LightweightCNN, MLP, CatBoost,
XGBoost, and RandomForest. Do not reorder them: `final.json` and each fold JSON
bind weight positions to this list.

## Split protocol

The inversion experiment uses five spatial outer folds. Complete
`track_group_id` values are assigned to folds within each region. For a given
outer fold, the Proposed classification coordinates define the shared held-out
anchor, and candidate training/validation rows within 500 m of that anchor are
marked `excluded_spatial_buffer`. A track-disjoint early-stopping set is then
selected from the remaining rows, with the same 500 m exclusion around
validation rows.

The authoritative row-level assignments are in
`splits/convexblend/row_splits.parquet`. The `source_row_index` column is the
zero-based row index in the run-specific split parquet. The five
`split_fold_<n>` columns preserve the exact state for every outer fold. The
`track_folds.csv` file is a smaller lookup for inspection, not a replacement
for the row-level artifact.

Photon classification uses fixed source-file partitions. Their filenames,
sizes, row counts, and content hashes are in
`splits/photon_classification/files.csv`.

## Weight distribution

All photon-classification checkpoints are regular repository files. Four final
ConvexBlend base checkpoints and the blend metadata are also regular files.
The complete final bundle is a GitHub Release asset because the compressed
random-forest checkpoint is approximately 500 MiB and exceeds GitHub's regular
100 MiB file limit. `models/WEIGHTS_MANIFEST.json` records asset and individual
file SHA-256 values. The downloader verifies the archive before extraction.

## Data availability boundary

Raw ATL03/ATL24 products, Sentinel-2 imagery, DEM rasters, and derived
full-scene GeoTIFFs are not duplicated in Git because they are large data
products rather than source code or split indices. The released split artifacts
identify the exact samples used. Users must obtain the underlying public remote
sensing products from their original providers or contact the authors for the
curated intermediate dataset.
