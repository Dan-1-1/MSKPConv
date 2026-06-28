"""E2: Ablation study — Table 2 of the paper.

Variants (all using STANDARD CROSS-ENTROPY, no physics-guided loss):
    1. w/o physical statistics (zero out 24-D features)
    2. Isotropic KPConv (replace strip kernels with circle kernels)
    3. Anisotropic KPConv (elliptical-kernel KPConv baseline)
    4. Single-scale strip (use only the smallest scale per multi-scale block)
    5. w/o track Transformer (bypass distance-biased global module)
    6. w/o gated decoder (replace ResidualChannelGate with identity)
    7. Full model

Final-evaluation behaviour matches E1: predictions are pooled per CSV across
``val`` + ``test`` (50 files), with KDE post-processing applied on top.

Output: ``<save_root>/e2_ablation/`` with summary, per-class, confusion matrices.
"""
import glob
import os

import torch

from .config import config_from_args, ExperimentConfig
from .dataset import build_loaders
from .losses import StandardCELoss
from .models.factory import build_baseline_model, build_proposed_wrapped
from .trainer import train_and_evaluate
from .utils.io import write_summary_table, write_per_class_metrics, write_confusion_matrices
from .utils.per_file_eval import collect_val_test_files, evaluate_on_files_pooled
from .utils.seed import set_seed


ABLATION_VARIANTS = [
    ("without_physical_stats",   "w_o_physical_statistics"),
    ("isotropic_kpconv",         "Isotropic_KPConv"),
    ("anisotropic_kpconv",       "Anisotropic_KPConv"),
    ("single_scale_strip",       "Single_scale_strip"),
    ("without_track_transformer","w_o_track_Transformer"),
    ("without_gated_decoder",    "w_o_gated_decoder"),
    ("full",                     "Full_model"),
]


def _final_eval(model, cfg: ExperimentConfig, device) -> dict:
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


def run_e2(cfg: ExperimentConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(cfg.save_root, "e2_ablation")
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders(cfg, input_mode="pos_local")
    print(f"[E2] train segs={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    results = {}
    for variant, display in ABLATION_VARIANTS:
        print(f"\n========== [E2] variant: {display} ==========")
        if variant == "anisotropic_kpconv":
            model = build_baseline_model(
                "anisotropic_kpconv",
                num_classes=cfg.num_classes,
                in_feat=cfg.local_feature_dims,
                pos_dim=2,
            )
        else:
            model = build_proposed_wrapped(cfg.source_root, variant=variant,
                                            num_classes=cfg.num_classes)
        criterion = StandardCELoss(num_classes=cfg.num_classes)  # NOTE: standard CE
        run = train_and_evaluate(
            model, train_loader, val_loader, test_loader,
            criterion, cfg, save_dir=os.path.join(save_dir, display),
        )

        pooled = _final_eval(run["model"], cfg, device)
        run["test_metrics"] = pooled["test_metrics"]
        run["test_summary_row"] = pooled["test_summary_row"]
        run["n_files_used"] = pooled.get("n_files_used", 0)
        run["n_files_total"] = pooled.get("n_files_total", 0)
        run["n_points"] = pooled.get("n_points", 0)

        results[display] = run
        print(f"[E2] {display} pooled summary: {run['test_summary_row']}")

    summary_rows = {name: r["test_summary_row"] for name, r in results.items()}
    write_summary_table(os.path.join(save_dir, "ablation_summary.csv"), summary_rows)
    write_per_class_metrics(os.path.join(save_dir, "per_class_metrics.csv"),
                             results, cfg.class_names)
    write_confusion_matrices(os.path.join(save_dir, "confusion_matrices"),
                              results, cfg.class_names)
    print(f"\n[E2] results written to {save_dir}")


def main():
    cfg = config_from_args()
    run_e2(cfg)


if __name__ == "__main__":
    main()
