"""Standalone evaluation: load pretrained checkpoints and run E1-style metrics.

For each trainable model (PointNet++/PointNeXt/KPConv/DGCNN/SKPConv):
    - Build the architecture
    - Load best.pt from ``<save_root>/e1_main_comparison/<display>/best.pt`` (or
      a user-provided override for SKPConv)
    - Run sliding-window vote inference on every CSV in val + test (50 files)
    - Apply KDE post-processing in physical metres (unless --no-postprocess)
    - Pool photons across files and compute metrics

For ATL24 reference:
    - Read CSVs from --atl24-eval-dir (default: /root/autodl-tmp/data/Test/Downloaded_ATL24)
    - Each CSV must contain BOTH 'Label' (ATL24 prediction) and 'labels'
      (manual ground truth) columns.
    - Pool photons across files and compute one set of metrics.

Output is written to ``<save_root>/e1_eval_only/`` (or --output-dir):
    metrics_summary.csv, per_class_metrics.csv, confusion_matrices/

Examples
--------
# Default: load checkpoints from save_root, post-process on, val+test pool
python -m baselines1.eval_pretrained \\
    --save-root /root/autodl-tmp/baselines1_results \\
    --atl24-eval-dir /root/autodl-tmp/data/Test/Downloaded_ATL24

# Use the user's pre-existing PhysicsKPConvNet checkpoint
python -m baselines1.eval_pretrained \\
    --save-root /root/autodl-tmp/baselines1_results \\
    --skpconv-ckpt /root/autodl-tmp/KPConv_new_GPU/Model_Checkpoints/best_model.pth \\
    --atl24-eval-dir /root/autodl-tmp/data/Test/Downloaded_ATL24

# Disable KDE post-processing to see raw model accuracy
python -m baselines1.eval_pretrained --no-postprocess
"""
import argparse
import glob
import os
from typing import Dict, Optional

import torch

from .config import build_argparser, ExperimentConfig
from .models.factory import build_baseline_model, build_proposed_wrapped
from .utils.atl24_reference import evaluate_atl24_reference
from .utils.io import (
    write_summary_table,
    write_per_class_metrics,
    write_confusion_matrices,
)
from .utils.per_file_eval import collect_val_test_files, evaluate_on_files_pooled
from .utils.seed import set_seed


MODEL_ENTRIES = [
    # (display name, factory key, default ckpt subdir)
    ("PointNet++",      "pointnet2", "PointNet++"),
    ("PointNeXt",       "pointnext", "PointNeXt"),
    ("KPConv",          "kpconv",    "KPConv"),
    ("DGCNN",           "dgcnn",     "DGCNN"),
    ("SKPConv",         "proposed",  "Proposed_method"),
]


def _build_model(key: str, cfg: ExperimentConfig) -> torch.nn.Module:
    if key == "proposed":
        return build_proposed_wrapped(
            cfg.source_root, variant="full", num_classes=cfg.num_classes
        )
    return build_baseline_model(
        key, num_classes=cfg.num_classes,
        in_feat=cfg.local_feature_dims, pos_dim=2,
    )


def _extract_state_dict(state):
    """Accept either a raw state_dict or a checkpoint dict; return the state_dict."""
    if isinstance(state, dict):
        for nested_key in ("model_state_dict", "state_dict", "model"):
            if nested_key in state and isinstance(state[nested_key], dict):
                return state[nested_key]
    return state


def _load_pretrained(model: torch.nn.Module, ckpt_path: str) -> bool:
    """Try strict load first, then fall back to non-strict. Returns success bool."""
    if not os.path.exists(ckpt_path):
        print(f"  [missing] {ckpt_path}")
        return False
    raw = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(raw)
    try:
        model.load_state_dict(state, strict=True)
        print(f"  loaded (strict): {ckpt_path}")
        return True
    except Exception as exc_strict:
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"  loaded (non-strict): {ckpt_path}")
            if missing:
                print(f"    missing keys: {len(missing)} (e.g. {list(missing)[:3]})")
            if unexpected:
                print(f"    unexpected keys: {len(unexpected)} (e.g. {list(unexpected)[:3]})")
            return True
        except Exception as exc_loose:
            print(f"  [load failed] strict err: {exc_strict}")
            print(f"  [load failed] loose  err: {exc_loose}")
            return False


def run_eval_only(cfg: ExperimentConfig,
                  atl24_eval_dir: str,
                  output_dir: str = "",
                  skpconv_ckpt: str = "",
                  ckpt_root: str = "") -> Dict[str, dict]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Default ckpt root mirrors E1 layout.
    ckpt_root = ckpt_root or os.path.join(cfg.save_root, "e1_main_comparison")

    # Eval files for trainable models.
    if cfg.eval_on_val_test:
        eval_files = collect_val_test_files(cfg)
    else:
        eval_files = sorted(glob.glob(os.path.join(cfg.test_dir, "*.csv")))
    print(f"[eval-only] {len(eval_files)} CSV(s) for model evaluation "
          f"(val+test={cfg.eval_on_val_test}, postprocess={cfg.apply_postprocess})")

    results: Dict[str, dict] = {}

    # 1) Trainable models
    for display, key, ckpt_subdir in MODEL_ENTRIES:
        print(f"\n========== {display} ==========")
        if key == "proposed" and skpconv_ckpt:
            ckpt_path = skpconv_ckpt
        else:
            ckpt_path = os.path.join(ckpt_root, ckpt_subdir, "best.pt")

        model = _build_model(key, cfg)
        if not _load_pretrained(model, ckpt_path):
            print(f"  [skip] {display}: cannot use this checkpoint")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        model.to(device).eval()
        pooled = evaluate_on_files_pooled(
            model, eval_files, cfg, device,
            apply_postprocess=cfg.apply_postprocess,
            surface_band_m=cfg.kde_surface_band_m,
            seabed_offset_m=cfg.kde_seabed_offset_m,
            kde_window_m=cfg.kde_window_m,
            kde_bw=cfg.kde_bw,
        )
        results[display] = pooled
        print(f"  summary: {pooled['test_summary_row']}")
        print(f"  pooled over {pooled.get('n_files_used', 0)} file(s), "
              f"{pooled.get('n_points', 0)} photon(s)")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2) ATL24 reference (CSVs must have both 'Label' and 'labels' columns)
    print(f"\n========== ATL24_reference ==========")
    print(f"  source dir: {atl24_eval_dir}")
    if not os.path.isdir(atl24_eval_dir):
        print(f"  [skip] directory does not exist: {atl24_eval_dir}")
    else:
        try:
            atl24 = evaluate_atl24_reference(
                test_dir=atl24_eval_dir,
                num_classes=cfg.num_classes,
                class_names=cfg.class_names,
            )
            results["ATL24_reference"] = atl24
            print(f"  summary: {atl24['test_summary_row']}")
            print(f"  files used: {atl24.get('n_files_used', 0)} / "
                  f"{atl24.get('n_files_total', 0)} | "
                  f"photons: {atl24.get('n_points', 0)}")
        except Exception as exc:
            print(f"  [ATL24 eval failed] {exc}")

    # 3) Write outputs
    out_dir = output_dir or os.path.join(cfg.save_root, "e1_eval_only")
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = {name: r["test_summary_row"] for name, r in results.items()}
    write_summary_table(os.path.join(out_dir, "metrics_summary.csv"), summary_rows)
    write_per_class_metrics(os.path.join(out_dir, "per_class_metrics.csv"),
                             results, cfg.class_names)
    write_confusion_matrices(os.path.join(out_dir, "confusion_matrices"),
                              results, cfg.class_names)
    print(f"\n[eval-only] results written to {out_dir}")
    return results


def main():
    parser = build_argparser()
    parser.add_argument(
        "--atl24-eval-dir",
        default="/root/autodl-tmp/data/Test/Downloaded_ATL24",
        help="Directory of CSVs containing 'Label' (ATL24) and 'labels' (manual GT) columns.",
    )
    parser.add_argument(
        "--output-dir", default="",
        help="Where to write metrics outputs. Default: <save_root>/e1_eval_only",
    )
    parser.add_argument(
        "--skpconv-ckpt", default="",
        help="Override checkpoint path for SKPConv. "
             "Useful for loading the user's pre-trained PhysicsKPConvNet weights.",
    )
    parser.add_argument(
        "--ckpt-root", default="",
        help="Root directory containing <model_dir>/best.pt for each baseline. "
             "Default: <save_root>/e1_main_comparison",
    )
    args = parser.parse_args()

    custom_keys = {"atl24_eval_dir", "output_dir", "skpconv_ckpt", "ckpt_root"}
    arg_dict = {k: v for k, v in vars(args).items() if k not in custom_keys}
    cfg = ExperimentConfig(**arg_dict).finalize()

    run_eval_only(
        cfg,
        atl24_eval_dir=args.atl24_eval_dir,
        output_dir=args.output_dir,
        skpconv_ckpt=args.skpconv_ckpt,
        ckpt_root=args.ckpt_root,
    )


if __name__ == "__main__":
    main()
