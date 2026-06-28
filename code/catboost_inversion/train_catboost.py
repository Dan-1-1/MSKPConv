"""CatBoost 训练 - 单区域

直接运行: python train.py
切换区域: 修改 config.py 中的 CURRENT_REGION
"""
import argparse
import json

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

import config
from utils import compute_metrics, load_region_csvs


def train_region(region: str = None) -> str:
    region = region or config.CURRENT_REGION
    print(f"\n{'='*60}\n[Train] Region = {region}\n{'='*60}")

    df = load_region_csvs(config.train_dir(region))
    df = df.dropna(subset=config.FEATURE_COLS + [config.TARGET_COL])
    print(f"Loaded {len(df):,} training rows")

    X = df[config.FEATURE_COLS].astype("float32").values
    y = df[config.TARGET_COL].astype("float32").values

    # Hold out 10% of Train CSV rows for CatBoost early stopping and validation.
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, random_state=42)

    model = CatBoostRegressor(**config.CATBOOST_PARAMS)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

    pred_va = model.predict(X_va)
    metrics = compute_metrics(y_va, pred_va)
    print(f"\n[Validation] R2={metrics['R2']:.3f} | RMSE={metrics['RMSE']:.3f} | "
          f"MAE={metrics['MAE']:.3f} | Bias={metrics['Bias']:+.3f}")

    out = config.output_dir(region)
    safe = config.safe_name(region)
    model_path = out / f"{safe}_catboost.cbm"
    model.save_model(str(model_path))
    print(f"Saved model: {model_path}")

    with open(out / "metrics_train.json", "w", encoding="utf-8") as f:
        json.dump({"region": region, "validation": metrics, "n_train": len(df)},
                  f, indent=2)

    importance = dict(zip(config.FEATURE_COLS,
                          [float(v) for v in model.get_feature_importance()]))
    with open(out / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(importance, f, indent=2)
    print(f"Feature importance: {importance}")

    return str(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default=None,
                        help="区域名，默认读取 config.CURRENT_REGION")
    parser.add_argument("--all", action="store_true",
                        help="训练所有 4 个区域")
    args = parser.parse_args()

    if args.all:
        for r in config.ALL_REGIONS:
            train_region(r)
    else:
        train_region(args.region)
