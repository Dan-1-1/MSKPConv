"""E4: 跨区域 CatBoost 迁移实验 (Leave-One-Region-Out)

训练: 其他 3 个区域全部数据
测试: 留出区域的 Test CSV + CUDEM 外部验证

输出:
  {region}/Catboost/e4_leave_one_out/
      {region}_LORO_predicted_depth.tif       LORO 模型整景预测
      {region}_LORO_predicted_depth.png       预测可视化
      {region}_LORO_cudem_scatter.png         CUDEM 散点图
      {region}_LORO_icesat2_scatter.png       ICESat-2 散点图
      {region}_LORO_result.json               结果 + 区域化对比

  {DATA_ROOT}/e4_LORO_summary.csv             4 折汇总表
"""
import csv
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

import config
from predict_validate import (
    cudem_validation,
    icesat2_validation,
    predict_full_scene_tif,
)
from utils import compute_metrics, load_region_csvs


def train_combined(train_regions):
    print(f"  Training on: {train_regions}")
    dfs = [load_region_csvs(config.train_dir(r)) for r in train_regions]
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=config.FEATURE_COLS + [config.TARGET_COL])
    print(f"  Total rows: {len(df):,}")

    X = df[config.FEATURE_COLS].astype("float32").values
    y = df[config.TARGET_COL].astype("float32").values
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, random_state=42)

    params = dict(config.CATBOOST_PARAMS)
    params["verbose"] = 200
    model = CatBoostRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    return model


def load_test(region):
    files = sorted(config.test_dir(region).glob("*.csv"))
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files],
                   ignore_index=True)
    return df.dropna(subset=config.FEATURE_COLS + [config.TARGET_COL]).reset_index(drop=True)


def load_regional_baseline(region):
    """读取该区域单独训练得到的 metrics.json"""
    path = config.output_dir(region) / "predictions" / "metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    summary_rows = []

    for held_out in config.ALL_REGIONS:
        train_regions = [r for r in config.ALL_REGIONS if r != held_out]
        print(f"\n{'='*60}\n[E4 LORO] Held-out = {held_out}\n{'='*60}")

        model = train_combined(train_regions)

        df_test = load_test(held_out)
        X = df_test[config.FEATURE_COLS].astype("float32").values
        df_test["Depth_Pred"] = model.predict(X).astype("float32")

        out = config.output_dir(held_out) / "e4_leave_one_out"
        out.mkdir(parents=True, exist_ok=True)
        safe = config.safe_name(held_out)

        # 散点图
        ic_png = out / f"{safe}_LORO_icesat2_scatter.png"
        ic_m = icesat2_validation(
            df_test, held_out, ic_png,
            title_suffix=f"LORO (train: {' + '.join(train_regions)})",
        )
        scene_id = config.prediction_scene_id(held_out)
        tif_path = out / f"{safe}_LORO_predicted_depth.tif"
        pred_tif = predict_full_scene_tif(model, held_out, scene_id, tif_path)
        if pred_tif is None:
            raise RuntimeError(f"Full-scene prediction failed for {held_out}; cannot run CUDEM validation")

        cu_png = out / f"{safe}_LORO_cudem_scatter.png"
        cu_m = cudem_validation(
            pred_tif, held_out, cu_png,
            title_suffix=f"LORO (train: {' + '.join(train_regions)})",
        )
        print(f"  ICESat-2: R2={ic_m['R2']:.3f} RMSE={ic_m['RMSE']:.3f} "
              f"MAE={ic_m['MAE']:.3f} Bias={ic_m['Bias']:+.3f}")
        print(f"  CUDEM:    R2={cu_m['R2']:.3f} RMSE={cu_m['RMSE']:.3f} "
              f"MAE={cu_m['MAE']:.3f} Bias={cu_m['Bias']:+.3f}")

        # 区域化对比
        regional = load_regional_baseline(held_out)
        result = {
            "held_out": held_out,
            "trained_on": train_regions,
            "icesat2": ic_m,
            "cudem": cu_m,
            "regional_baseline": regional,
        }
        with open(out / f"{safe}_LORO_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        # 汇总表行
        row = {
            "Held_Out": held_out,
            "Trained_On": " + ".join(train_regions),
            "ICESat2_R2": round(ic_m["R2"], 4),
            "ICESat2_RMSE": round(ic_m["RMSE"], 4),
            "ICESat2_MAE": round(ic_m["MAE"], 4),
            "ICESat2_Bias": round(ic_m["Bias"], 4),
            "CUDEM_R2": round(cu_m["R2"], 4),
            "CUDEM_RMSE": round(cu_m["RMSE"], 4),
            "CUDEM_MAE": round(cu_m["MAE"], 4),
            "CUDEM_Bias": round(cu_m["Bias"], 4),
        }
        if regional:
            r_ic = regional.get("icesat2_test", {})
            r_cu = regional.get("cudem_validation", {})
            row["Regional_ICESat2_RMSE"] = round(r_ic.get("RMSE", float("nan")), 4)
            row["Regional_CUDEM_RMSE"] = round(r_cu.get("RMSE", float("nan")), 4)
            if r_ic.get("RMSE") is not None:
                row["Delta_ICESat2_RMSE"] = round(ic_m["RMSE"] - r_ic["RMSE"], 4)
            if r_cu.get("RMSE") is not None:
                row["Delta_CUDEM_RMSE"] = round(cu_m["RMSE"] - r_cu["RMSE"], 4)
        summary_rows.append(row)

    # 写汇总 CSV
    summary_path = config.DATA_ROOT / "e4_LORO_summary.csv"
    if summary_rows:
        keys = sorted({k for row in summary_rows for k in row.keys()},
                      key=lambda x: list(summary_rows[0].keys()).index(x)
                      if x in summary_rows[0] else 99)
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"\n[E4] Summary written: {summary_path}")

        # 控制台汇总打印
        print("\n=== E4 LORO Summary ===")
        for r in summary_rows:
            print(f"{r['Held_Out']:<14} | ICESat2 RMSE={r['ICESat2_RMSE']:.3f} | "
                  f"CUDEM RMSE={r['CUDEM_RMSE']:.3f}")


if __name__ == "__main__":
    main()
