"""Same-support comparison of learned, empirical, and constrained-blend regressors."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
from experiment_1_main.workflow import load_config, metrics, write_json  # noqa: E402


def load_selection_config(path: Path | None = None) -> dict[str, Any]:
    path = path or HERE / "config.yaml"
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def selection_root(main_cfg: dict[str, Any], run_id: str, cfg: dict[str, Any]) -> Path:
    root = Path(main_cfg["output_root"]) / run_id / "model_selection" / str(cfg["experiment_id"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def selection_model_root(main_cfg: dict[str, Any], run_id: str, cfg: dict[str, Any]) -> Path:
    """Store the two newly fitted empirical models beside the main checkpoints."""
    root = Path(main_cfg.get("model_output_root", main_cfg["output_root"])) / run_id / "model_selection" / str(cfg["experiment_id"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def main_model_root(main_cfg: dict[str, Any], run_id: str, method: str) -> Path:
    """Return the already fitted main-experiment model directory."""
    return Path(main_cfg.get("model_output_root", main_cfg["output_root"])) / run_id / "models" / "pooled" / method


def _split_path(main_cfg: dict[str, Any], run_id: str, region: str, method: str) -> Path:
    return Path(main_cfg["output_root"]) / run_id / "splits" / region / f"{method}.parquet"


def _finite_rows(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return the shared model support after excluding NaN and infinite values."""
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    keep = np.isfinite(numeric.to_numpy(dtype="float64")).all(axis=1)
    clean = frame.loc[keep].copy()
    clean.loc[:, columns] = numeric.loc[keep]
    return clean


def qc(main_cfg: dict[str, Any], cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    root = selection_root(main_cfg, run_id, cfg); checks = []; errors = []
    for region in main_cfg["regions"]:
        rid = str(region["id"])
        for method in cfg["methods"]:
            p = _split_path(main_cfg, run_id, rid, str(method)); item = {"region_id": rid, "method": str(method), "exists": p.exists()}
            if not p.exists(): errors.append(str(p)); checks.append(item); continue
            d = pd.read_parquet(p); item["rows"] = len(d); item["groups"] = int(d.track_group_id.nunique())
            for fold in cfg["folds"]:
                col = f"split_fold_{int(fold)}"
                if col not in d: errors.append(f"{p}: missing {col}")
                else: item[str(fold)] = d[col].value_counts().to_dict()
            checks.append(item)
    base_names = [str(x) for x in cfg["convex_blend"]["base_regressors"]]
    for fold in cfg["folds"]:
        anchor = Path(main_cfg["output_root"]) / run_id / "evaluation" / "cudem_anchor" / f"fold_{int(fold)}_samples.parquet"
        model_root = main_model_root(main_cfg, run_id, str(cfg["methods"][0]))
        required = [model_root / "ConvexBlend" / f"fold_{int(fold)}.json"]
        required.extend(
            model_root / name / (f"fold_{int(fold)}.cbm" if name == "CatBoost" else f"fold_{int(fold)}.joblib")
            for name in base_names
        )
        missing = [str(path) for path in [anchor, *required] if not path.exists()]
        checks.append({"fold": int(fold), "anchor": str(anchor), "main_models": len(required), "missing": missing})
        errors.extend(missing)
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors, "checks": checks}
    write_json(root / "qc" / "summary.json", result)
    if errors: raise RuntimeError("model-selection QC failed: " + "; ".join(errors[:5]))
    return result


class LyzengaRegressor:
    """Linear SDB model on log(B02), log(B03), and log(B04)."""
    def __init__(self, eps: float = 1e-6):
        from sklearn.linear_model import LinearRegression
        self.eps = float(eps); self.model = LinearRegression()

    def _transform(self, x):
        z = np.asarray(x, dtype="float64")[:, :3]
        return np.log(np.maximum(z, self.eps))

    def fit(self, x, y): self.model.fit(self._transform(x), y); return self
    def predict(self, x): return self.model.predict(self._transform(x))


class StumpfLogRatioRegressor:
    """Linear SDB model on log(1000*B02)/log(1000*B03)."""
    def __init__(self, scale: float = 1000.0, eps: float = 1e-6):
        from sklearn.linear_model import LinearRegression
        self.scale = float(scale); self.eps = float(eps); self.model = LinearRegression()

    def _transform(self, x):
        z = np.asarray(x, dtype="float64")
        blue = np.maximum(z[:, 0], self.eps); green = np.maximum(z[:, 1], self.eps)
        ratio = np.log(self.scale * blue + self.eps) / np.log(self.scale * green + self.eps)
        return ratio.reshape(-1, 1)

    def fit(self, x, y): self.model.fit(self._transform(x), y); return self
    def predict(self, x): return self.model.predict(self._transform(x))


def _scene_dir(main_cfg: dict[str, Any], region_id: str, scene_id: str) -> Path:
    """Resolve a Sentinel-2 scene directory from the main experiment config."""
    region = next((x for x in main_cfg["regions"] if str(x["id"]) == str(region_id)), None)
    if region is None:
        raise KeyError(f"Unknown region_id for image patch extraction: {region_id}")
    return Path(main_cfg["data_root"]) / str(region["s2_root"]) / str(scene_id)


def _read_sentinel2_stack(scene_dir: Path) -> np.ndarray:
    """Read the four raw Sentinel-2 bands used by the image CNN.

    The prepared tabular experiment derives eight values at the matched pixel.
    This function deliberately reads the raw B02/B03/B04/B08 image stack so
    the CNN receives a spatial tensor rather than a vector of derived values.
    """
    import rasterio

    stack_path = scene_dir / "B02_B03_B04_B08_Stack.tif"
    if stack_path.exists():
        with rasterio.open(stack_path) as src:
            data = src.read(indexes=[1, 2, 3, 4], out_dtype="float32")
    else:
        bands = [scene_dir / f"{band}.tif" for band in ("B02", "B03", "B04", "B08")]
        if not all(x.exists() for x in bands):
            missing = [str(x) for x in bands if not x.exists()]
            raise FileNotFoundError(f"Sentinel-2 raw bands missing under {scene_dir}: {missing}")
        band_arrays = []
        for band_path in bands:
            with rasterio.open(band_path) as src:
                band_arrays.append(src.read(1, out_dtype="float32"))
        data = np.stack(band_arrays, axis=0)
    if data.ndim != 3 or data.shape[0] != 4:
        raise ValueError(f"Expected a four-band Sentinel-2 stack, got {data.shape} from {scene_dir}")
    return data


def extract_sentinel2_patches(frame: pd.DataFrame, main_cfg: dict[str, Any], patch_size: int = 11) -> np.ndarray:
    """Extract raw B02/B03/B04/B08 image windows centered at matched pixels.

    Rows are grouped by region and scene so each raster is read once per call.
    Border windows are reflect-padded. Non-finite or cloud-masked pixels are
    replaced by the finite median of their local band, preserving the center
    pixel while avoiding NaNs in the CNN tensor.
    """
    import rasterio

    size = int(patch_size)
    if size < 3 or size % 2 == 0:
        raise ValueError("image_patch_size must be an odd integer >= 3")
    rows = frame.reset_index(drop=True)
    output = np.empty((len(rows), 4, size, size), dtype="float32")
    radius = size // 2
    for (region_id, scene_id), group in rows.groupby(["region_id", "scene_id"], sort=False):
        scene_dir = _scene_dir(main_cfg, str(region_id), str(scene_id))
        stack = _read_sentinel2_stack(scene_dir)
        cloud = None
        cloud_path = scene_dir / "cloud_mask.tif"
        if cloud_path.exists():
            with rasterio.open(cloud_path) as src:
                cloud = src.read(1)
        pad_mode = "reflect" if stack.shape[1] > 1 and stack.shape[2] > 1 else "edge"
        padded = np.pad(stack, ((0, 0), (radius, radius), (radius, radius)), mode=pad_mode)
        for idx in group.index.to_numpy(dtype=int):
            row = int(round(float(rows.at[idx, "img_row"])))
            col = int(round(float(rows.at[idx, "img_col"])))
            if row < 0 or col < 0 or row >= stack.shape[1] or col >= stack.shape[2]:
                raise IndexError(f"Image coordinate outside scene bounds: {scene_id} ({row}, {col})")
            patch = padded[:, row:row + size, col:col + size].copy()
            if cloud is not None:
                mask = cloud[max(0, row - radius):row + radius + 1, max(0, col - radius):col + radius + 1]
                if mask.shape == (size, size):
                    patch[:, mask != 0] = np.nan
            for band in range(patch.shape[0]):
                values = patch[band]
                finite = np.isfinite(values)
                fill = float(np.nanmedian(values)) if finite.any() else 0.0
                values[~finite] = fill
                patch[band] = values
            output[idx] = patch
    return output


class LightweightCNNRegressor:
    """Small 2-D CNN regressor operating directly on Sentinel-2 image patches."""

    def __init__(self, conv_channels=(16, 32), kernel_size=3, dense_units=16,
                 dropout=0.10, learning_rate=0.001, weight_decay=0.0001,
                 batch_size=2048, epochs=150, patience=20, min_delta=1e-5,
                 validation_fraction=0.10, device="cpu", num_threads=4,
                 random_seed=42):
        self.conv_channels = tuple(int(x) for x in conv_channels)
        self.kernel_size = int(kernel_size); self.dense_units = int(dense_units)
        self.dropout = float(dropout); self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay); self.batch_size = int(batch_size)
        self.epochs = int(epochs); self.patience = int(patience); self.min_delta = float(min_delta)
        self.validation_fraction = float(validation_fraction); self.device = str(device)
        self.num_threads = int(num_threads); self.random_seed = int(random_seed)
        self.network_ = None; self.channel_mean_ = None; self.channel_scale_ = None

    def _make_network(self):
        import torch.nn as nn

        c1, c2 = self.conv_channels
        pad = self.kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(4, c1, self.kernel_size, padding=pad), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, self.kernel_size, padding=pad), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(c2, self.dense_units), nn.ReLU(inplace=True),
            nn.Dropout(self.dropout), nn.Linear(self.dense_units, 1),
        )

    def fit(self, x, y, eval_set=None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.set_num_threads(max(1, self.num_threads))
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        x = np.asarray(x, dtype="float32"); y = np.asarray(y, dtype="float32").reshape(-1)
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"LightweightCNNRegressor expects [N,4,H,W], got {x.shape}")
        self.channel_mean_ = x.mean(axis=(0, 2, 3), dtype="float64").astype("float32")
        self.channel_scale_ = x.std(axis=(0, 2, 3), dtype="float64").astype("float32")
        self.channel_scale_[self.channel_scale_ < 1e-6] = 1.0
        x = (x - self.channel_mean_[None, :, None, None]) / self.channel_scale_[None, :, None, None]

        if eval_set is None:
            rng = np.random.default_rng(self.random_seed)
            order = rng.permutation(len(x)); n_val = max(1, int(round(len(x) * self.validation_fraction)))
            val_idx, train_idx = order[:n_val], order[n_val:]
            x_val, y_val = x[val_idx], y[val_idx]
            x_fit, y_fit = x[train_idx], y[train_idx]
        else:
            x_fit, y_fit = x, y
            x_val, y_val = eval_set
            x_val = (np.asarray(x_val, dtype="float32") - self.channel_mean_[None, :, None, None]) / self.channel_scale_[None, :, None, None]
            y_val = np.asarray(y_val, dtype="float32").reshape(-1)

        device = "cuda" if self.device.lower() in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
        model = self._make_network().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()
        gen = torch.Generator().manual_seed(self.random_seed)
        loader = DataLoader(TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit[:, None])),
                            batch_size=self.batch_size, shuffle=True, generator=gen, num_workers=0)
        x_val_t = torch.from_numpy(x_val).to(device); y_val_t = torch.from_numpy(y_val[:, None]).to(device)
        best_loss = float("inf"); best_state = None; stale = 0; self.best_epoch_ = 0
        for epoch in range(self.epochs):
            model.train()
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device); optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(x_val_t), y_val_t).item())
            if val_loss < best_loss - self.min_delta:
                best_loss = val_loss; stale = 0; self.best_epoch_ = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience: break
        if best_state is not None: model.load_state_dict(best_state)
        self.network_ = model.to("cpu").eval()
        self.device_ = "cpu"
        self.n_features_in_ = 4
        return self

    def fit_full(self, x, y, epochs: int):
        """Fit every supplied sample for a fixed CV-selected epoch count."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.set_num_threads(max(1, self.num_threads))
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        x = np.asarray(x, dtype="float32"); y = np.asarray(y, dtype="float32").reshape(-1)
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(f"LightweightCNNRegressor expects [N,4,H,W], got {x.shape}")
        self.channel_mean_ = x.mean(axis=(0, 2, 3), dtype="float64").astype("float32")
        self.channel_scale_ = x.std(axis=(0, 2, 3), dtype="float64").astype("float32")
        self.channel_scale_[self.channel_scale_ < 1e-6] = 1.0
        x = (x - self.channel_mean_[None, :, None, None]) / self.channel_scale_[None, :, None, None]
        device = "cuda" if self.device.lower() in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
        model = self._make_network().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()
        generator = torch.Generator().manual_seed(self.random_seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(y[:, None])),
            batch_size=self.batch_size, shuffle=True, generator=generator, num_workers=0,
        )
        self.best_epoch_ = max(1, int(epochs))
        for _ in range(self.best_epoch_):
            model.train()
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device); optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb); loss.backward(); optimizer.step()
        self.network_ = model.to("cpu").eval(); self.device_ = "cpu"; self.n_features_in_ = 4
        return self

    def predict(self, x):
        import torch

        if self.network_ is None: raise RuntimeError("LightweightCNNRegressor is not fitted")
        x = np.asarray(x, dtype="float32")
        x = (x - self.channel_mean_[None, :, None, None]) / self.channel_scale_[None, :, None, None]
        with torch.no_grad():
            return self.network_(torch.from_numpy(x)).numpy().reshape(-1)


def _models(cfg: dict[str, Any], seed: int):
    from catboost import CatBoostRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    cb = dict(cfg["catboost"]); cb["random_seed"] = int(seed)
    cnn_cfg = dict(cfg["lightweight_cnn"])
    cnn_cfg.pop("image_patch_size", None)
    models = {
        "CatBoost": CatBoostRegressor(**cb, verbose=False, eval_metric="RMSE", od_type="Iter"),
        "XGBoost": XGBRegressor(**dict(cfg["xgboost"]), random_state=int(seed), n_jobs=4),
        "RandomForest": RandomForestRegressor(**dict(cfg["random_forest"]), random_state=int(seed)),
        "MLP": make_pipeline(StandardScaler(), MLPRegressor(**dict(cfg["mlp"]), random_state=int(seed))),
        "LightweightCNN": LightweightCNNRegressor(**cnn_cfg, random_seed=int(seed)),
        "Lyzenga": LyzengaRegressor(**dict(cfg.get("lyzenga", {}))),
        "StumpfLogRatio": StumpfLogRatioRegressor(**dict(cfg.get("stumpf_log_ratio", {}))),
    }
    return models


def _load_base_test_predictions(root: Path, base_names: list[str], method: str, fold: int) -> pd.DataFrame:
    """Merge same-support base predictions for one untouched outer test fold."""
    key_cols = ["region_id", "method", "fold", "track_group_id", "scene_id", "img_row", "img_col"]
    merged = None
    reference = None
    for base_name in base_names:
        path = root / "predictions" / "pooled" / method / base_name / f"fold_{fold}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing base prediction for convex blend: {path}")
        frame = pd.read_parquet(path)
        required = key_cols + ["Depth", "CUDEM_Depth", "Prediction", "Latitude", "Longitude"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing convex-blend columns: {missing}")
        if frame.duplicated(key_cols).any():
            raise ValueError(f"duplicate test coordinates in {path}")
        if reference is None:
            reference = frame
        values = frame[key_cols + ["Depth", "CUDEM_Depth", "Prediction"]].rename(
            columns={"Depth": f"Depth__{base_name}", "CUDEM_Depth": f"CUDEM_Depth__{base_name}",
                     "Prediction": f"Prediction__{base_name}"}
        )
        merged = values if merged is None else merged.merge(values, on=key_cols, how="inner", validate="one_to_one")

    depth_cols = [f"Depth__{name}" for name in base_names]
    depth_matrix = merged[depth_cols].to_numpy(dtype="float64")
    if not np.allclose(depth_matrix, depth_matrix[:, [0]], rtol=0, atol=1e-8, equal_nan=False):
        raise ValueError("base models do not share identical test depths")
    cudem_cols = [f"CUDEM_Depth__{name}" for name in base_names]
    cudem_matrix = merged[cudem_cols].to_numpy(dtype="float64")
    if not np.allclose(cudem_matrix, cudem_matrix[:, [0]], rtol=0, atol=1e-8, equal_nan=False):
        raise ValueError("base models do not share identical CUDEM test depths")
    result = merged[key_cols].copy()
    result["Depth"] = depth_matrix[:, 0]
    result["CUDEM_Depth"] = cudem_matrix[:, 0]
    for base_name in base_names:
        result[f"Prediction__{base_name}"] = merged[f"Prediction__{base_name}"].to_numpy(dtype="float64")
    metadata = reference[key_cols + ["Latitude", "Longitude"]]
    result = result.merge(metadata, on=key_cols, how="left", validate="one_to_one")
    columns = key_cols + ["Latitude", "Longitude", "Depth", "CUDEM_Depth"] + [f"Prediction__{name}" for name in base_names]
    return result[columns]


def _saved_model_prediction(model_root: Path, cfg: dict[str, Any], main_cfg: dict[str, Any], name: str, fold: int, frame: pd.DataFrame, features: list[str], cnn_patches: np.ndarray | None = None) -> np.ndarray:
    """Predict rows with a base model already fitted by the main experiment."""
    from joblib import load

    model_dir = model_root / name
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(model_dir / f"fold_{fold}.cbm"))
    else:
        model = load(model_dir / f"fold_{fold}.joblib")
    if name == "LightweightCNN":
        patches = cnn_patches
        if patches is None:
            patches = extract_sentinel2_patches(frame, main_cfg, int(cfg["lightweight_cnn"].get("image_patch_size", 11)))
        return np.asarray(model.predict(patches), dtype="float64")
    return np.asarray(model.predict(frame[features]), dtype="float64")


def _fit_convex_weights(y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Fit non-negative weights summing to one on validation predictions only."""
    from scipy.optimize import minimize

    n_models = predictions.shape[1]
    initial = np.full(n_models, 1.0 / n_models, dtype="float64")
    objective = lambda weights: float(np.mean((np.asarray(y) - predictions @ weights) ** 2))
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.x).all():
        return initial
    weights = np.clip(np.asarray(result.x, dtype="float64"), 0.0, 1.0)
    return weights / weights.sum()


def run(main_cfg: dict[str, Any], cfg: dict[str, Any], run_id: str, max_tasks: int | None = None) -> dict[str, Any]:
    from joblib import dump
    root = selection_root(main_cfg, run_id, cfg); features = list(main_cfg["features"])
    fitted_model_root = selection_model_root(main_cfg, run_id, cfg)
    metrics_dir = root / "metrics"; metrics_dir.mkdir(parents=True, exist_ok=True)
    rows = []; tasks = 0; seed = int(cfg["random_seed"])
    method_ids = [str(x) for x in cfg["methods"]]
    reusable = set(str(x) for x in cfg["convex_blend"]["base_regressors"])
    for method in method_ids:
        parts = []
        for region in main_cfg["regions"]:
            p = _split_path(main_cfg, run_id, str(region["id"]), method)
            if p.exists(): parts.append(pd.read_parquet(p))
        if not parts: continue
        d = pd.concat(parts, ignore_index=True)
        for fold in cfg["folds"]:
            col = f"split_fold_{int(fold)}"
            required = features + ["Depth"]
            train = _finite_rows(d[(d.fold != int(fold)) & d[col].eq("train")], required)
            val = _finite_rows(d[(d.fold != int(fold)) & d[col].eq("validation")], required)
            anchor_path = Path(main_cfg["output_root"]) / run_id / "evaluation" / "cudem_anchor" / f"fold_{int(fold)}_samples.parquet"
            if not anchor_path.exists():
                raise FileNotFoundError(f"main-experiment CUDEM anchor is not ready: {anchor_path}")
            test = _finite_rows(pd.read_parquet(anchor_path), required + ["CUDEM_Depth"])
            if len(train) < 5 or val.empty or test.empty: continue
            Xtr, ytr = train[features], train.Depth
            Xte, yte, yte_cudem = test[features], test.Depth, test.CUDEM_Depth
            main_models = main_model_root(main_cfg, run_id, method)
            cnn_patches = None
            for name in cfg["regressors"]:
                out_path = root / "predictions" / "pooled" / method / str(name) / f"fold_{fold}.parquet"
                if cfg.get("save_predictions", True) and out_path.exists():
                    saved = pd.read_parquet(out_path)
                    if len(saved) == len(test) and {"Prediction", "CUDEM_Depth"}.issubset(saved.columns):
                        pred = saved["Prediction"].to_numpy(float)
                        mm = metrics(saved["CUDEM_Depth"].to_numpy(float), pred); mm["abs_bias"] = abs(float(mm["bias"]))
                        imm = metrics(saved["Depth"].to_numpy(float), pred)
                        rows.append({"scope": "pooled", "method": method, "fold": int(fold), "regressor": str(name), "status": "OK",
                                     "primary_reference": "CUDEM", **{f"icesat2_{k}": v for k, v in imm.items()}, **mm})
                        for rid, rg in saved.groupby("region_id"):
                            rmm = metrics(rg.CUDEM_Depth.to_numpy(float), rg.Prediction.to_numpy(float))
                            rows.append({"scope": "region", "region_id": rid, "method": method, "fold": int(fold), "regressor": str(name), "status": "OK", **rmm})
                        tasks += 1
                        pd.DataFrame(rows).to_csv(metrics_dir / "fold_metrics.csv", index=False)
                        print(f"pooled selection {tasks}: reused {method}/{name}/fold{fold} test={len(saved)}", flush=True)
                        if max_tasks is not None and tasks >= max_tasks:
                            return {"tasks": tasks}
                        continue
                if str(name) in reusable:
                    if str(name) == "LightweightCNN" and cnn_patches is None:
                        cnn_patches = extract_sentinel2_patches(
                            test, main_cfg, int(cfg["lightweight_cnn"].get("image_patch_size", 11))
                        )
                    pred = _saved_model_prediction(
                        main_models, cfg, main_cfg, str(name), int(fold), test, features, cnn_patches
                    )
                    action = "reused-main-model"
                    model = None
                elif str(name) == "ConvexBlend":
                    base_names = [str(x) for x in cfg["convex_blend"]["base_regressors"]]
                    base_test = _load_base_test_predictions(root, base_names, method, int(fold))
                    blend_path = main_models / "ConvexBlend" / f"fold_{int(fold)}.json"
                    blend_info = json.loads(blend_path.read_text(encoding="utf-8"))
                    weights = np.asarray(blend_info["weights"], dtype="float64")
                    prediction_matrix = base_test[[f"Prediction__{base_name}" for base_name in base_names]].to_numpy(dtype="float64")
                    pred = prediction_matrix @ weights
                    out = base_test[["region_id", "method", "fold", "track_group_id", "scene_id", "img_row", "img_col", "Latitude", "Longitude", "Depth", "CUDEM_Depth"]].copy()
                    out["Prediction"] = pred
                    mm = metrics(out["CUDEM_Depth"].to_numpy(dtype=float), pred); mm["abs_bias"] = abs(float(mm["bias"]))
                    imm = metrics(out["Depth"].to_numpy(dtype=float), pred)
                    rows.append({"scope": "pooled", "method": method, "fold": int(fold), "regressor": "ConvexBlend", "status": "OK",
                                 "primary_reference": "CUDEM", **{f"icesat2_{k}": v for k, v in imm.items()}, **mm})
                    for rid, rg in out.groupby("region_id"):
                        rmm = metrics(rg.CUDEM_Depth.to_numpy(float), rg.Prediction.to_numpy(float))
                        rows.append({"scope": "region", "region_id": rid, "method": method, "fold": int(fold), "regressor": "ConvexBlend", "status": "OK", **rmm})
                    if cfg.get("save_predictions", True):
                        out_path.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(out_path, index=False)
                    model_path = fitted_model_root / "models" / "pooled" / method / "ConvexBlend" / f"fold_{fold}.json"
                    write_json(model_path, {"type": "reused_main_convex_blend", "source": str(blend_path), "base_regressors": base_names, "weights": weights.tolist(), "objective": blend_info.get("objective", "rmse"), "fit_split": blend_info.get("fit_split", "outer_fold_validation"), "fold": int(fold), "method": method, "validation_rows": blend_info.get("validation_rows", len(val))})
                    tasks += 1
                    pd.DataFrame(rows).to_csv(metrics_dir / "fold_metrics.csv", index=False)
                    print(f"pooled selection {tasks}: constructed {method}/ConvexBlend/fold{fold} weights={np.round(weights, 4).tolist()} test={len(out)}", flush=True)
                    if max_tasks is not None and tasks >= max_tasks:
                        return {"tasks": tasks}
                    continue
                else:
                    model = _models(cfg, seed)[str(name)]
                    model.fit(Xtr, ytr)
                    pred = np.asarray(model.predict(Xte), dtype=float)
                    action = "fitted-new-model"
                mm = metrics(yte_cudem.to_numpy(float), pred); mm["abs_bias"] = abs(float(mm["bias"]))
                imm = metrics(yte.to_numpy(float), pred)
                rows.append({"scope": "pooled", "method": method, "fold": int(fold), "regressor": str(name), "status": "OK",
                             "primary_reference": "CUDEM", **{f"icesat2_{k}": v for k, v in imm.items()}, **mm}); tasks += 1
                for rid, rg in test.assign(Prediction=pred).groupby("region_id"):
                    rmm = metrics(rg.CUDEM_Depth.to_numpy(float), rg.Prediction.to_numpy(float)); rows.append({"scope": "region", "region_id": rid, "method": method, "fold": int(fold), "regressor": str(name), "status": "OK", **rmm})
                if cfg.get("save_predictions", True):
                    out = test[["region_id", "method", "fold", "track_group_id", "scene_id", "img_row", "img_col", "Latitude", "Longitude", "Depth", "CUDEM_Depth"]].copy(); out["Prediction"] = pred
                    out_path.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(out_path, index=False)
                if cfg.get("save_models", True) and model is not None:
                    model_path = fitted_model_root / "models" / "pooled" / method / str(name) / f"fold_{fold}"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    if str(name) == "CatBoost": model.save_model(str(model_path.with_suffix(".cbm")))
                    else: dump(model, model_path.with_suffix(".joblib"), compress=3)
                # Persist a restartable checkpoint after each completed task.
                pd.DataFrame(rows).to_csv(metrics_dir / "fold_metrics.csv", index=False)
                print(f"pooled selection {tasks}: {action} {method}/{name}/fold{fold} train={len(train)} val={len(val)} test={len(test)}", flush=True)
                if max_tasks is not None and tasks >= max_tasks:
                    return {"tasks": tasks}
    df = pd.DataFrame(rows); df.to_csv(metrics_dir / "fold_metrics.csv", index=False)
    write_json(root / "run_status.json", {"status": "completed", "training_mode": "pooled", "tasks": tasks})
    return {"tasks": tasks}


def report(main_cfg: dict[str, Any], cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    root = selection_root(main_cfg, run_id, cfg); p = root / "metrics" / "fold_metrics.csv"
    (root / "figures").mkdir(parents=True, exist_ok=True)
    if not p.exists(): raise FileNotFoundError(p)
    df = pd.read_csv(p); metrics_to_report = ["rmse", "mae", "median_ae", "bias", "abs_bias", "r2", "pearson_r2"]
    pooled = df[(df.scope == "pooled") & df.status.eq("OK")].copy(); region = df[(df.scope == "region") & df.status.eq("OK")].copy()
    support = pooled.pivot(index="fold", columns="regressor", values="n")
    if support.isna().any().any() or not support.nunique(axis=1).eq(1).all():
        raise RuntimeError("regressors do not share identical CUDEM test support")
    grouped = pooled.groupby(["method", "regressor"], as_index=False)[metrics_to_report].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(x).strip("_") for x in grouped.columns.to_flat_index()]; grouped.to_csv(root / "metrics" / "summary.csv", index=False)
    sig = []
    for method, g in pooled.groupby("method"):
        pivot = g.pivot(index="fold", columns="regressor", values="rmse")
        if "ConvexBlend" not in pivot: continue
        for other in [x for x in cfg["regressors"] if x != "ConvexBlend" and x in pivot]:
            pair = pivot[["ConvexBlend", other]].dropna(); diff = pair[other] - pair["ConvexBlend"]
            if len(diff) >= 2 and np.any(diff != 0):
                stat, pv = wilcoxon(diff.to_numpy(float), alternative="two-sided", method="auto")
            else: stat, pv = np.nan, np.nan
            sig.append({"scope": "pooled", "method": method, "comparison": f"{other} vs ConvexBlend", "n_folds": len(diff), "mean_rmse_difference": float(diff.mean()) if len(diff) else np.nan, "wilcoxon_stat": stat, "p_value": pv})
    sig_df = pd.DataFrame(sig)
    if not sig_df.empty:
        sig_df["p_value_bonferroni"] = np.minimum(sig_df["p_value"] * len(sig_df), 1.0)
    sig_df.to_csv(root / "metrics" / "significance_vs_convexblend.csv", index=False)
    if not region.empty:
        rg = region.groupby(["region_id", "method", "regressor"], as_index=False)[metrics_to_report].agg(["mean", "std"]).reset_index()
        rg.columns = ["_".join(x).strip("_") for x in rg.columns.to_flat_index()]; rg.to_csv(root / "metrics" / "region_summary.csv", index=False)
    weight_rows = []
    for method in cfg["methods"]:
        for fold in cfg["folds"]:
            weight_path = selection_model_root(main_cfg, run_id, cfg) / "models" / "pooled" / str(method) / "ConvexBlend" / f"fold_{int(fold)}.json"
            if not weight_path.exists():
                continue
            weight_item = json.loads(weight_path.read_text(encoding="utf-8"))
            row = {"method": str(method), "fold": int(fold), "validation_rows": weight_item.get("validation_rows"), "objective": weight_item.get("objective", "rmse")}
            row.update({f"weight_{name}": float(weight) for name, weight in zip(weight_item["base_regressors"], weight_item["weights"])})
            weight_rows.append(row)
    if weight_rows:
        pd.DataFrame(weight_rows).sort_values(["method", "fold"]).to_csv(root / "metrics" / "convex_blend_weights.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for ax, metric, label in zip(axes, ["rmse", "mae", "r2"], ["RMSE (m)", "MAE (m)", "R²"]):
        x = np.arange(len(grouped)); ax.bar(x, grouped[f"{metric}_mean"], yerr=grouped[f"{metric}_std"].fillna(0), capsize=2)
        ax.set_xticks(x, [str(r) for r in grouped.regressor], rotation=50, ha="right"); ax.set_ylabel(label); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Pooled ten-region regressor comparison on identical CUDEM support"); fig.savefig(root / "figures" / "pooled_regressor_comparison.png", dpi=220); plt.close(fig)
    write_json(root / "run_status.json", {"status": "reported", "training_mode": "pooled", "rows": int(len(df))})
    return {"rows": len(df), "pooled_rows": len(pooled), "region_rows": len(region), "significance": str(root / "metrics" / "significance_vs_convexblend.csv")}
