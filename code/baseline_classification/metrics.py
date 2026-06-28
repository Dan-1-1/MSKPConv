"""Evaluation metrics for photon classification."""
from typing import Dict, List, Sequence
import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    mask = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    yt = y_true[mask]
    yp = y_pred[mask]
    np.add.at(cm, (yt, yp), 1)
    return cm


def per_class_metrics(cm: np.ndarray) -> Dict[str, np.ndarray]:
    eps = 1e-10
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    support = cm.sum(axis=1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "support": support,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: Sequence[str],
) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    pcm = per_class_metrics(cm)
    total = cm.sum()
    oa = float(np.diag(cm).sum() / max(total, 1))
    miou = float(np.mean(pcm["iou"]))
    macro_f1 = float(np.mean(pcm["f1"]))
    out = {
        "OA": oa,
        "mIoU": miou,
        "Macro-F1": macro_f1,
        "confusion_matrix": cm,
        "per_class": {
            name: {
                "precision": float(pcm["precision"][i]),
                "recall": float(pcm["recall"][i]),
                "f1": float(pcm["f1"][i]),
                "iou": float(pcm["iou"][i]),
                "support": int(pcm["support"][i]),
            }
            for i, name in enumerate(class_names)
        },
    }
    return out


def metrics_summary_row(metrics: Dict[str, object], class_names: Sequence[str]) -> Dict[str, float]:
    """Return the standardized columns matching paper Tables 2/3."""
    seabed = metrics["per_class"].get("seabed", {})
    return {
        "OA": round(metrics["OA"], 4),
        "mIoU": round(metrics["mIoU"], 4),
        "Macro-F1": round(metrics["Macro-F1"], 4),
        "Seabed_P": round(seabed.get("precision", 0.0), 4),
        "Seabed_R": round(seabed.get("recall", 0.0), 4),
        "Seabed_F1": round(seabed.get("f1", 0.0), 4),
        "Seabed_IoU": round(seabed.get("iou", 0.0), 4),
    }
