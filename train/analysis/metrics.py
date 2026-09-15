"""Data computation for the analysis charts - each function returns a plain,
JSON/CSV-serializable structure, saved alongside the PNG that renders it."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def compute_confusion(df: pd.DataFrame) -> dict[str, int]:
    counts = df["Classification"].value_counts().to_dict()
    return {key: int(counts.get(key, 0)) for key in ("TP", "FN", "FP", "TN")}


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metrics_from_confusion(counts: dict[str, int]) -> dict[str, float]:
    tp, fn, fp, tn = counts["TP"], counts["FN"], counts["FP"], counts["TN"]
    total = tp + fn + fp + tn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "total": total,
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "miss_rate": _safe_div(fn, tp + fn),
        "false_alarm_rate": _safe_div(fp, fp + tn),
    }


def compute_roc_pr(df: pd.DataFrame) -> dict[str, Any] | None:
    """ROC/PR curve points for a split. None if the split has only one class."""
    y_true = df["GT_Label"].to_numpy()
    if len(np.unique(y_true)) < 2:
        return None
    y_score = df["Anomaly_Score"].to_numpy()
    fpr, tpr, roc_thresh = roc_curve(y_true, y_score)
    precision, recall, pr_thresh = precision_recall_curve(y_true, y_score)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_thresholds": roc_thresh.tolist(),
        "roc_auc": float(auc(fpr, tpr)),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "pr_thresholds": pr_thresh.tolist(),
    }


def compute_threshold_sweep(df: pd.DataFrame, max_points: int = 60) -> pd.DataFrame:
    scores = df["Anomaly_Score"].dropna()
    if scores.empty:
        return pd.DataFrame()
    values = np.sort(scores.unique())
    if len(values) > max_points:
        values = np.quantile(values, np.linspace(0, 1, max_points))
    pad = max(float(np.std(values)) * 0.05, 1e-9)
    grid = np.unique(np.r_[values[0] - pad, values, values[-1] + pad])

    rows = []
    gt = df["GT_Label"].to_numpy()
    scores_arr = df["Anomaly_Score"].to_numpy()
    for threshold in grid:
        pred = (scores_arr >= threshold).astype(int)
        tp = int(np.sum((pred == 1) & (gt == 1)))
        fn = int(np.sum((pred == 0) & (gt == 1)))
        fp = int(np.sum((pred == 1) & (gt == 0)))
        tn = int(np.sum((pred == 0) & (gt == 0)))
        row = {"threshold": float(threshold)}
        row.update(metrics_from_confusion({"TP": tp, "FN": fn, "FP": fp, "TN": tn}))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def compute_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    anomalous = df[df["GT_Label"] == 1].copy()
    if anomalous.empty:
        return pd.DataFrame()
    grouped = anomalous.groupby("Defect_Type")["Classification"].value_counts().unstack(fill_value=0)
    for col in ("TP", "FN"):
        if col not in grouped:
            grouped[col] = 0
    grouped["support"] = grouped["TP"] + grouped["FN"]
    grouped["recall"] = grouped["TP"] / grouped["support"].replace(0, np.nan)
    grouped["miss_rate"] = grouped["FN"] / grouped["support"].replace(0, np.nan)
    return grouped.reset_index().sort_values(["miss_rate", "support"], ascending=[False, False])
