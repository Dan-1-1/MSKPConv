# Experiment 6: Proposed-chain uncertainty budget

This module quantifies the chain used by the final bathymetry product:

`Proposed photon classification -> sea surface -> refraction -> tide -> ConvexBlend inversion`.

The classification evidence is restricted to the manually labelled independent
test set. It reports OA, mIoU, and Macro-F1 from the three prescribed seeds and
never uses `ATL24_Label`. These dimensionless classification scores are not
converted to metres. Consequently, the metric uncertainty budget combines only
the measured/controlled metric components; the residual component is labelled
`inversion_and_unresolved` and includes interactions that cannot be uniquely
identified from the available experiments.

The total error is measured from spatially isolated five-fold ConvexBlend OOF
predictions against CUDEM. Bias is reported separately. Component variances are
not claimed to be causally independent. NOAA station comparisons are de-biased
before their robust standard deviation is calculated. Refraction is a controlled
parameter sensitivity over the tested refractive-index range, not a probability
distribution.

Run with the geospatial environment:

```powershell
conda run -n env1 python scripts/ten_region_bathymetry/experiment_6_uncertainty_budget/run.py `
  --run-id 20261011T000000Z_selected_r2
```

Large result files are written to the configured `G:` output root. The generated
GeoTIFFs are CUDEM-conditional empirical predictive uncertainty layers, not
theoretical lower bounds.

