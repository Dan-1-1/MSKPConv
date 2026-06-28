"""Helpers for writing summary CSVs."""
import csv
import os
from typing import Dict, Sequence


def write_summary_table(out_path: str,
                        rows: Dict[str, Dict[str, float]],
                        column_order: Sequence[str] = ("OA", "mIoU", "Macro-F1",
                                                        "Seabed_P", "Seabed_R",
                                                        "Seabed_F1", "Seabed_IoU")) -> None:
    """rows[name] -> {col: value}; outputs a CSV with one row per name."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + list(column_order))
        for name, vals in rows.items():
            writer.writerow([name] + [vals.get(c, "") for c in column_order])


def write_per_class_metrics(out_path: str,
                             results: Dict[str, Dict[str, object]],
                             class_names: Sequence[str]) -> None:
    """Write per-class precision/recall/F1/IoU for every model."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Class", "Precision", "Recall", "F1", "IoU", "Support"])
        for name, res in results.items():
            metrics = res["test_metrics"]
            for cn in class_names:
                pc = metrics["per_class"].get(cn, {})
                writer.writerow([
                    name, cn,
                    round(pc.get("precision", 0.0), 4),
                    round(pc.get("recall", 0.0), 4),
                    round(pc.get("f1", 0.0), 4),
                    round(pc.get("iou", 0.0), 4),
                    pc.get("support", 0),
                ])


def write_confusion_matrices(out_dir: str,
                              results: Dict[str, Dict[str, object]],
                              class_names: Sequence[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, res in results.items():
        cm = res["test_metrics"]["confusion_matrix"]
        path = os.path.join(out_dir, f"cm_{name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([""] + list(class_names))
            for i, row in enumerate(cm):
                writer.writerow([class_names[i]] + list(row))
