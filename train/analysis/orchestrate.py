"""Top-level orchestration: wires metrics.py's computations to charts.py's
rendering for every split, then for the whole run. Called once from
train.pipeline after all test/val/contamination passes complete.

Runs once per train invocation. Everything here is derived from data already
collected by the callbacks (predictions, per-image pixel stats, epoch metrics)
- no extra model inference happens in this module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import charts
from .metrics import (
    compute_category_breakdown,
    compute_confusion,
    compute_roc_pr,
    compute_threshold_sweep,
    metrics_from_confusion,
)

logger = logging.getLogger("train_script")


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def generate_split_analysis(
    analysis_dir: Path,
    figures_dir: Path,
    split: str,
    df: pd.DataFrame,
    threshold: float | None,
    per_image_stats: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute + chart everything derivable from one split's predictions.

    Returns a small summary dict (auroc, aupr, etc.) for inclusion elsewhere.
    Never raises - any failure is logged and skipped so a plotting bug can
    never take down an expensive training run.
    """
    split_key = split.lower()
    summary: dict[str, Any] = {}

    def safe(name: str, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[analysis] Skipping {name} ({split}): {exc}")

    confusion = compute_confusion(df) if not df.empty else {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
    summary["confusion"] = confusion
    summary.update(metrics_from_confusion(confusion))

    roc_data = None
    try:
        roc_data = compute_roc_pr(df) if not df.empty else None
        if roc_data is not None:
            summary["auroc"] = roc_data["roc_auc"]
            _write_json(
                {k: v for k, v in roc_data.items() if k not in ("fpr", "tpr", "precision", "recall")} | {
                    "fpr": roc_data["fpr"], "tpr": roc_data["tpr"],
                    "precision": roc_data["precision"], "recall": roc_data["recall"],
                },
                analysis_dir / f"roc_pr_curve_{split_key}.json",
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[analysis] Skipping ROC/PR computation ({split}): {exc}")

    safe("score distribution chart", charts.chart_score_distribution, df, threshold, figures_dir / f"score_distribution_{split_key}.png")

    if split_key == "contamination":
        safe("contamination summary chart", charts.chart_contamination_summary, confusion, threshold, figures_dir / "contamination_flagged_summary.png")
    else:
        safe("roc chart", charts.chart_roc_curve, roc_data, figures_dir / f"roc_curve_{split_key}.png")
        safe("pr chart", charts.chart_pr_curve, roc_data, figures_dir / f"pr_curve_{split_key}.png")
        safe("confusion matrix chart", charts.chart_confusion_matrix, confusion, figures_dir / f"confusion_matrix_{split_key}.png", f"Confusion matrix ({split})", threshold)

        try:
            sweep_df = compute_threshold_sweep(df) if not df.empty else pd.DataFrame()
            if not sweep_df.empty:
                sweep_df.to_csv(analysis_dir / f"threshold_sweep_{split_key}.csv", index=False)
            safe("threshold sweep chart", charts.chart_threshold_sweep, sweep_df, threshold, figures_dir / f"threshold_sweep_{split_key}.png")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[analysis] Skipping threshold sweep ({split}): {exc}")

        try:
            breakdown_df = compute_category_breakdown(df) if not df.empty else pd.DataFrame()
            if not breakdown_df.empty:
                breakdown_df.to_csv(analysis_dir / f"category_breakdown_{split_key}.csv", index=False)
            safe("defect breakdown chart", charts.chart_defect_breakdown, breakdown_df, figures_dir / f"defect_breakdown_{split_key}.png")
            safe("defect score spread chart", charts.chart_defect_score_spread, df, threshold, figures_dir / f"defect_score_spread_{split_key}.png")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[analysis] Skipping defect breakdown ({split}): {exc}")

        if per_image_stats is not None and not per_image_stats.empty:
            safe("pixel score distribution chart", charts.chart_pixel_score_distribution, per_image_stats, figures_dir / f"pixel_score_distribution_{split_key}.png")
            safe("iou/dice distribution chart", charts.chart_iou_dice_distribution, per_image_stats, figures_dir / f"iou_dice_distribution_{split_key}.png")
            safe("defect size vs score chart", charts.chart_defect_size_vs_score, per_image_stats, figures_dir / f"defect_size_vs_score_{split_key}.png")

            try:
                if "GT_Mask_Available" in per_image_stats.columns:
                    masked = per_image_stats[per_image_stats["GT_Mask_Available"]]
                else:
                    masked = pd.DataFrame()
                if not masked.empty:
                    pixel_gt = masked["GT_Label"].to_numpy()
                    if len(np.unique(pixel_gt)) >= 2:
                        # True per-pixel ROC/PR would need every raw anomaly_map/gt_mask held in
                        # memory at once; the aggregate IoU/Dice + peak-score views above cover
                        # that ground instead. This confusion matrix is at the image level,
                        # restricted to images that actually have a ground-truth mask.
                        pixel_confusion = {
                            "TP": int(((masked["Classification"] == "TP")).sum()),
                            "FN": int(((masked["Classification"] == "FN")).sum()),
                            "FP": int(((masked["Classification"] == "FP")).sum()),
                            "TN": int(((masked["Classification"] == "TN")).sum()),
                        }
                        safe(
                            "pixel confusion matrix chart",
                            charts.chart_pixel_confusion_matrix,
                            pixel_confusion,
                            figures_dir / f"pixel_confusion_matrix_{split_key}.png",
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[analysis] Skipping pixel confusion matrix ({split}): {exc}")

    return summary


def generate_run_level_analysis(
    analysis_dir: Path,
    figures_dir: Path,
    split_counts: dict[str, dict[str, int]],
    epoch_df: pd.DataFrame | None,
    resource_df: pd.DataFrame | None = None,
) -> None:
    """Charts that exist once per run, not once per split."""
    try:
        charts.chart_split_composition(split_counts, figures_dir / "split_composition.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[analysis] Skipping split composition chart: {exc}")

    try:
        charts.chart_epoch_progression_metrics(epoch_df if epoch_df is not None else pd.DataFrame(), figures_dir / "epoch_progression_metrics.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[analysis] Skipping epoch progression (metrics) chart: {exc}")

    try:
        charts.chart_epoch_progression_loss(epoch_df if epoch_df is not None else pd.DataFrame(), figures_dir / "epoch_progression_loss.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[analysis] Skipping epoch progression (loss) chart: {exc}")

    try:
        charts.chart_resource_usage(resource_df if resource_df is not None else pd.DataFrame(), figures_dir / "resource_usage.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[analysis] Skipping resource usage chart: {exc}")


def run_post_test_analysis(
    output_path: Path,
    predictions_by_split: dict[str, pd.DataFrame],
    thresholds_by_split: dict[str, float | None],
    per_image_stats_by_split: dict[str, pd.DataFrame],
    split_counts: dict[str, dict[str, int]],
    epoch_df: pd.DataFrame | None,
    resource_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Top-level entry point called from train.pipeline after all test passes finish.

    Wrapped in try/except at the call site as well - this function itself is
    defensive per-split, but a hard failure here must never fail the run.
    """
    analysis_dir = output_path / "analysis"
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}
    for split, df in predictions_by_split.items():
        summaries[split] = generate_split_analysis(
            analysis_dir,
            figures_dir,
            split,
            df,
            thresholds_by_split.get(split),
            per_image_stats_by_split.get(split),
        )

    generate_run_level_analysis(analysis_dir, figures_dir, split_counts, epoch_df, resource_df)

    _write_json(summaries, analysis_dir / "summary.json")
    return summaries
