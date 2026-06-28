"""E1: Main comparison — Table 3 of the paper.

All six entries:
    PointNet++, PointNeXt, KPConv, DGCNN, ATL24 reference, SKPConv.

Loss: standard cross-entropy for ALL trainable models.

Final-evaluation behaviour
--------------------------
After training (with early-stopping on val Macro-F1), each model is evaluated
**per CSV** over the union of ``val`` and ``test`` directories (50 files):
predictions and the manual ``labels`` column are pooled across all photons,
then a single set of metrics is computed (no per-file averaging).

A KDE-based sea-surface refinement is applied on top of each model's raw
predictions (in physical metres):
    - signal photons within ±0.5 m of the estimated surface  -> surface
    - signal photons more than 1 m below the surface         -> seabed

The ATL24 reference still uses ``test_dir`` only and is evaluated by pooling
the ``Label`` (ATL24) and ``labels`` (manual GT) columns from each test CSV.

Output: ``<save_root>/e1_main_comparison/`` with summary, per-class, confusion
matrices, and best.pt for each trainable model.
"""
import glob
import os

import torch

from .config import config_from_args, ExperimentConfig
from .dataset import build_loaders
from .losses import StandardCELoss
from .models.factory import build_baseline_model, build_proposed_wrapped
from .trainer import train_and_evaluate
from .utils.atl24_reference import evaluate_atl24_reference
from .utils.io import write_summary_table, write_per_class_metrics, write_confusion_matrices
from .utils.per_file_eval import collect_val_test_files, evaluate_on_files_pooled
from .utils.seed import set_seed


TRAINABLE_MODELS = [
    ("pointnet2", "PointNet++"),
    ("pointnext", "PointNeXt"),
    ("kpconv", "KPConv"),
    ("dgcnn", "DGCNN"),
    ("proposed", "SKPConv"),
]


def _final_eval(model, cfg: ExperimentConfig, device) -> dict:
    """Run the per-file val+test pooled evaluation (with optional KDE post-process)."""
    if cfg.eval_on_val_test:
        eval_files = collect_val_test_files(cfg)
    else:
        eval_files = sorted(glob.glob(os.path.join(cfg.test_dir, "*.csv")))
    print(f"      pooled eval over {len(eval_files)} file(s) "
          f"(val+test={cfg.eval_on_val_test}, postprocess={cfg.apply_postprocess})")
    return evaluate_on_files_pooled(
        model, eval_files, cfg, device,
        apply_postprocess=cfg.apply_postprocess,
        surface_band_m=cfg.kde_surface_band_m,
        seabed_offset_m=cfg.kde_seabed_offset_m,
        kde_window_m=cfg.kde_window_m,
        kde_bw=cfg.kde_bw,
    )


def run_e1(cfg: ExperimentConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(cfg.save_root, "e1_main_comparison")
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders(cfg, input_mode="pos_local")
    print(f"[E1] train segs={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    results = {}

    # 1) Trainable models
    for key, display in TRAINABLE_MODELS:
        print(f"\n========== [E1] training {display} ==========")
        if key == "proposed":
            model = build_proposed_wrapped(cfg.source_root, variant="full",
                                            num_classes=cfg.num_classes)
        else:
            model = build_baseline_model(key, num_classes=cfg.num_classes,
                                          in_feat=cfg.local_feature_dims, pos_dim=2)
        criterion = StandardCELoss(num_classes=cfg.num_classes)
        run = train_and_evaluate(
            model, train_loader, val_loader, test_loader,
            criterion, cfg, save_dir=os.path.join(save_dir, display),
        )

        # Override the loader-based test metrics with per-file pooled eval over
        # val+test (with KDE post-processing applied to raw model predictions).
        pooled = _final_eval(run["model"], cfg, device)
        run["test_metrics"] = pooled["test_metrics"]
        run["test_summary_row"] = pooled["test_summary_row"]
        run["n_files_used"] = pooled.get("n_files_used", 0)
        run["n_files_total"] = pooled.get("n_files_total", 0)
        run["n_points"] = pooled.get("n_points", 0)

        results[display] = run
        print(f"[E1] {display} pooled summary: {run['test_summary_row']}")

    # 2) ATL24 reference (no training, test_dir only, no post-processing)
    print("\n========== [E1] evaluating ATL24 reference ==========")
    try:
        atl24_res = evaluate_atl24_reference(
            test_dir=cfg.test_dir,
            num_classes=cfg.num_classes, class_names=cfg.class_names,
        )
        results["ATL24_reference"] = atl24_res
        print(f"[E1] ATL24 reference summary: {atl24_res['test_summary_row']}")
    except Exception as exc:
        print(f"[E1] ATL24 reference failed: {exc}")

    # Write outputs
    summary_rows = {name: r["test_summary_row"] for name, r in results.items()}
    write_summary_table(os.path.join(save_dir, "metrics_summary.csv"), summary_rows)
    write_per_class_metrics(os.path.join(save_dir, "per_class_metrics.csv"),
                             results, cfg.class_names)
    write_confusion_matrices(os.path.join(save_dir, "confusion_matrices"),
                              results, cfg.class_names)
    print(f"\n[E1] results written to {save_dir}")


def main():
    cfg = config_from_args()
    run_e1(cfg)


if __name__ == "__main__":
    main()
