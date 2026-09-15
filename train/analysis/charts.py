"""Chart rendering. Every chart is a static PNG (matplotlib + seaborn) backed by
a JSON/CSV file (see metrics.py / orchestrate.py) containing the exact numbers
plotted, so a future cross-model comparison tool can read the numbers directly
without ever decoding a PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .palette import (
    CATEGORICAL,
    CHROME,
    SEQUENTIAL_CMAP,
    SPLIT_ORDER,
    STATUS,
    class_color_map,
    empty_chart,
    new_fig,
    relative_luminance,
    save,
    style_axes,
)


def _hist_kde_combo(
    ax: plt.Axes,
    normal: pd.Series,
    anomalous: pd.Series,
    xlabel: str,
    bins: int = 40,
) -> None:
    """Filled histogram (thin dark outline) + KDE line on top, one series each.

    Density-normalized so the histogram and the KDE line share one real axis -
    a raw-count histogram and a density curve can't be overlaid meaningfully on
    the same axis otherwise, and this method never uses a second y-axis.
    """
    colors = (CATEGORICAL[0], CATEGORICAL[7])
    non_empty = [s for s in (normal, anomalous) if not s.empty]
    if non_empty:
        combined = np.concatenate([s.to_numpy() for s in non_empty])
        lo, hi = combined.min(), combined.max()
        edges = np.linspace(lo, hi, bins + 1) if hi > lo else np.linspace(lo - 0.5, hi + 0.5, bins + 1)
    else:
        edges = bins

    for series, color, name in ((normal, colors[0], "Normal"), (anomalous, colors[1], "Anomalous")):
        if series.empty:
            continue
        ax.hist(series, bins=edges, density=True, alpha=0.4, color=color, edgecolor=CHROME["ink"], linewidth=0.8, label=name)
        if series.nunique() > 1:
            sns.kdeplot(series, ax=ax, color=color, linewidth=2.6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")


def chart_score_distribution(df: pd.DataFrame, threshold: float | None, path: Path) -> None:
    if df.empty:
        empty_chart(path)
        return
    normal = df.loc[df["GT_Label"] == 0, "Anomaly_Score"].dropna()
    anomalous = df.loc[df["GT_Label"] == 1, "Anomaly_Score"].dropna()
    if normal.empty and anomalous.empty:
        empty_chart(path)
        return
    fig, ax = new_fig((7, 4.5))
    _hist_kde_combo(ax, normal, anomalous, "Anomaly score (raw)")
    if threshold is not None and np.isfinite(threshold):
        ax.axvline(threshold, color=CHROME["ink"], linestyle="--", linewidth=1.5, label="Threshold")
    ax.set_title("Score distribution")
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=CHROME["ink_secondary"])
    save(fig, path)


def chart_roc_curve(roc_data: dict[str, Any] | None, path: Path) -> None:
    if roc_data is None:
        empty_chart(path, "ROC undefined - split has only one class")
        return
    fig, ax = new_fig((5.5, 5.5))
    ax.plot(roc_data["fpr"], roc_data["tpr"], color=CATEGORICAL[0], linewidth=2, label="ROC")
    ax.plot([0, 1], [0, 1], color=CHROME["muted"], linewidth=1.2, linestyle="--", label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve (AUROC={roc_data['roc_auc']:.3f})")
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=CHROME["ink_secondary"])
    save(fig, path)


def chart_pr_curve(roc_data: dict[str, Any] | None, path: Path) -> None:
    if roc_data is None:
        empty_chart(path, "PR curve undefined - split has only one class")
        return
    fig, ax = new_fig((5.5, 5.5))
    ax.plot(roc_data["recall"], roc_data["precision"], color=CATEGORICAL[0], linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    style_axes(ax)
    save(fig, path)


def chart_confusion_matrix(counts: dict[str, int], path: Path, title: str = "Confusion matrix") -> None:
    tn, fp, fn, tp = counts["TN"], counts["FP"], counts["FN"], counts["TP"]
    z = np.array([[tn, fp], [fn, tp]])
    z_norm = z / max(z.max(), 1)
    fig, ax = new_fig((5, 4.5))
    ax.imshow(z_norm, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred. Normal", "Pred. Anomalous"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True Normal", "True Anomalous"])
    ax.tick_params(colors=CHROME["ink_secondary"], length=0)
    # Text color follows each cell's actual luminance - white flips in on dark
    # cells, dark ink on light cells - instead of one fixed color that goes
    # unreadable once a cell gets dark enough.
    for i in range(2):
        for j in range(2):
            luminance = relative_luminance(SEQUENTIAL_CMAP(z_norm[i, j]))
            text_color = "#ffffff" if luminance < 0.6 else CHROME["ink"]
            ax.text(j, i, str(int(z[i, j])), ha="center", va="center", fontsize=16, fontweight="bold", color=text_color)
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_facecolor(CHROME["surface"])
    save(fig, path)


def chart_contamination_summary(counts: dict[str, int], path: Path) -> None:
    """Contamination pool is entirely 'normal' by label - show flagged vs clean."""
    flagged = counts.get("FP", 0)
    clean = counts.get("TN", 0)
    total = flagged + clean
    if total == 0:
        empty_chart(path)
        return
    fig, ax = new_fig((7, 2.4))
    ax.barh([0], [clean], color=STATUS["good"], label="Clean (below threshold)")
    ax.barh([0], [flagged], left=[clean], color=STATUS["critical"], label="Flagged (above threshold)")
    pct = 100 * flagged / total
    ax.set_yticks([])
    ax.set_xlim(0, total * 1.02)
    ax.set_title(f"Contamination scan: {flagged}/{total} training-pool images flagged as anomalous ({pct:.1f}%)")
    style_axes(ax, grid_axis="x")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, labelcolor=CHROME["ink_secondary"])
    save(fig, path)


def chart_threshold_sweep(sweep_df: pd.DataFrame, current_threshold: float | None, path: Path) -> None:
    if sweep_df.empty:
        empty_chart(path)
        return
    fig, ax = new_fig((7.5, 5))
    metric_cols = ["precision", "recall", "f1", "false_alarm_rate", "miss_rate"]
    for color, col in zip(CATEGORICAL, metric_cols):
        ax.plot(sweep_df["threshold"], sweep_df[col], color=color, linewidth=2, label=col.replace("_", " "))
    if current_threshold is not None and np.isfinite(current_threshold):
        ax.axvline(current_threshold, color=CHROME["ink"], linestyle="--", linewidth=1.2, label="Operating threshold")
    ax.set_xlabel("Threshold (raw score)")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.02)
    ax.set_title("Metrics vs. threshold")
    style_axes(ax)
    ax.legend(frameon=False, ncol=2, fontsize=9, labelcolor=CHROME["ink_secondary"])
    save(fig, path)


def chart_defect_breakdown(breakdown_df: pd.DataFrame, path: Path) -> None:
    if breakdown_df.empty:
        empty_chart(path, "No anomalous images in this split")
        return
    plot_df = breakdown_df.sort_values("miss_rate", ascending=True).reset_index(drop=True)
    fig, ax = new_fig((7.5, max(3.5, 0.45 * len(plot_df) + 1)))
    y = np.arange(len(plot_df))
    height = 0.38
    xmax = max(plot_df["TP"].max(), plot_df["FN"].max(), 1)
    bars_tp = ax.barh(y + height / 2, plot_df["TP"], height=height, color=STATUS["good"], label="Caught (TP)")
    bars_fn = ax.barh(y - height / 2, plot_df["FN"], height=height, color=STATUS["critical"], label="Missed (FN)")
    for bars, values in ((bars_tp, plot_df["TP"]), (bars_fn, plot_df["FN"])):
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(
                    bar.get_width() + xmax * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(value)),
                    va="center",
                    fontsize=9,
                    color=CHROME["ink_secondary"],
                )
    ax.set_xlim(0, xmax * 1.22)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Defect_Type"])
    ax.set_xlabel("Image count")
    ax.set_title("Defect breakdown, worst first (by miss rate)")
    style_axes(ax, grid_axis="x")
    ax.legend(frameon=False, labelcolor=CHROME["ink_secondary"])
    save(fig, path)


def chart_defect_score_spread(df: pd.DataFrame, path: Path) -> None:
    anomalous_types = sorted(df.loc[df["GT_Label"] == 1, "Defect_Type"].dropna().unique())
    if not anomalous_types:
        empty_chart(path, "No anomalous images in this split")
        return

    spread_df = df.copy()
    spread_df["Row_Label"] = np.where(spread_df["GT_Label"] == 0, "Good", spread_df["Defect_Type"])
    color_map = class_color_map(anomalous_types)
    color_map["Good"] = STATUS["good"]

    medians = spread_df.groupby("Row_Label")["Anomaly_Score"].median().sort_values()
    order = list(medians.index)
    if "Good" in order:
        order = ["Good"] + [row for row in order if row != "Good"]

    fig, ax = new_fig((7.5, max(3.5, 0.5 * len(order) + 1)))
    # Muted box for the quartile summary, individual score dots on top colored
    # per class - shows both the summary stats and exactly where scores land.
    sns.boxplot(
        data=spread_df,
        x="Anomaly_Score",
        y="Row_Label",
        order=order,
        ax=ax,
        color=CHROME["gridline"],
        fliersize=0,
        linewidth=1,
        boxprops=dict(alpha=0.5),
        whiskerprops=dict(alpha=0.5),
    )
    sns.stripplot(
        data=spread_df,
        x="Anomaly_Score",
        y="Row_Label",
        order=order,
        ax=ax,
        hue="Row_Label",
        legend=False,
        palette=[color_map[row] for row in order],
        size=5,
        alpha=0.85,
        jitter=0.25,
    )
    ax.set_xlabel("Anomaly score (raw)")
    ax.set_ylabel("")
    ax.set_title("Score spread by defect type")
    style_axes(ax, grid_axis="x")
    save(fig, path)


def chart_split_composition(split_counts: dict[str, dict[str, int]], path: Path) -> None:
    """split_counts: {'Train': {'normal': n, 'anomalous': n}, 'Validation': {...}, ...}"""
    splits = [s for s in SPLIT_ORDER if s in split_counts]
    if not splits:
        empty_chart(path)
        return
    normal_vals = [split_counts[s].get("normal", 0) for s in splits]
    anom_vals = [split_counts[s].get("anomalous", 0) for s in splits]
    fig, ax = new_fig((6.5, 4))
    x = np.arange(len(splits))
    ax.bar(x, normal_vals, color=STATUS["good"], label="Normal")
    ax.bar(x, anom_vals, bottom=normal_vals, color=STATUS["critical"], label="Anomalous")
    for i, (n, a) in enumerate(zip(normal_vals, anom_vals)):
        if n + a > 0:
            ax.text(i, n + a + max(normal_vals + anom_vals) * 0.01, str(n + a), ha="center", color=CHROME["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Image count")
    ax.set_title("Dataset split composition")
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=CHROME["ink_secondary"])
    save(fig, path)


def chart_epoch_progression(epoch_df: pd.DataFrame, path: Path) -> None:
    if epoch_df.empty or "epoch" not in epoch_df.columns:
        empty_chart(path, "No epoch metrics recorded")
        return
    metric_cols = [c for c in epoch_df.columns if c != "epoch" and epoch_df[c].notna().any()]
    if not metric_cols:
        empty_chart(path, "No epoch metrics recorded")
        return
    fig, ax = new_fig((7, 4.5))
    n_epochs = epoch_df["epoch"].nunique()
    for color, col in zip(CATEGORICAL, metric_cols):
        marker = "o" if n_epochs <= 5 else None
        ax.plot(epoch_df["epoch"], epoch_df[col], color=color, linewidth=2, marker=marker, label=col)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    title = "Training curve"
    if n_epochs == 1:
        title += " (single-pass model - one epoch recorded)"
    ax.set_title(title)
    style_axes(ax)
    ax.legend(frameon=False, ncol=2, fontsize=9, labelcolor=CHROME["ink_secondary"])
    save(fig, path)


def chart_resource_usage(resource_df: pd.DataFrame, path: Path) -> None:
    """CPU cores / RAM / GPU usage over elapsed wall-clock time, shaded by pipeline
    stage (fit/test/val/contamination). resource_monitor.py already samples on a
    wall-clock timer, so this works the same whether the run was a single-epoch
    stat model or a many-epoch gradient-trained one.

    CPU cores and RAM are this job's own process (meaningful on a shared node);
    GPU utilization is NVML's device-wide reading (meaningful when the job has
    exclusive access to its GPU, the SLURM norm here) - each gets its own panel
    since they're different units, not overlaid on one "percent" axis.
    """
    panel_specs = [
        ("cpu_cores_used", "CPU cores (this process)", CATEGORICAL[0]),
        ("ram_used_gb", "RAM (GB, this process)", CATEGORICAL[2]),
        ("gpu_util_percent", "GPU util % (device-wide)", CATEGORICAL[1]),
    ]
    available = [spec for spec in panel_specs if resource_df is not None and not resource_df.empty
                 and spec[0] in resource_df.columns and resource_df[spec[0]].notna().any()]
    if not available:
        empty_chart(path, "No resource samples recorded")
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(7.5, 2.6 * len(available)), sharex=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(CHROME["surface"])

    stages = list(dict.fromkeys(resource_df["stage"]))
    stage_colors = {stage: CATEGORICAL[i % len(CATEGORICAL)] for i, stage in enumerate(stages)}

    for ax, (col, label, color) in zip(axes, available):
        for stage in stages:
            stage_df = resource_df[resource_df["stage"] == stage]
            ax.axvspan(stage_df["elapsed_sec"].min(), stage_df["elapsed_sec"].max(), color=stage_colors[stage], alpha=0.08, lw=0)
        ax.plot(resource_df["elapsed_sec"], resource_df[col], color=color, linewidth=1.8)
        ax.set_ylabel(label)
        style_axes(ax)

    axes[-1].set_xlabel("Elapsed time (s)")
    axes[0].set_title("Resource usage over time")
    save(fig, path)


def chart_pixel_score_distribution(stats_df: pd.DataFrame, path: Path) -> None:
    if stats_df.empty:
        empty_chart(path)
        return
    normal = stats_df.loc[stats_df["GT_Label"] == 0, "Max_Score_In_Map"].dropna()
    anomalous = stats_df.loc[stats_df["GT_Label"] == 1, "Max_Score_In_Map"].dropna()
    if normal.empty and anomalous.empty:
        empty_chart(path)
        return
    fig, ax = new_fig((7, 4.5))
    _hist_kde_combo(ax, normal, anomalous, "Max in-map anomaly score (normalized)")
    ax.set_title("Pixel-level peak score distribution")
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=CHROME["ink_secondary"])
    save(fig, path)


def chart_iou_dice_distribution(stats_df: pd.DataFrame, path: Path) -> None:
    subset = stats_df.dropna(subset=["IoU", "Dice"]) if "IoU" in stats_df.columns else pd.DataFrame()
    if subset.empty:
        empty_chart(path, "No ground-truth masks available for this split")
        return
    fig, ax = new_fig((6, 4.5))
    data = pd.DataFrame({"IoU": subset["IoU"], "Dice": subset["Dice"]})
    melted = data.melt(var_name="Metric", value_name="Score")
    sns.boxplot(
        data=melted,
        x="Metric",
        y="Score",
        hue="Metric",
        legend=False,
        ax=ax,
        palette=[CATEGORICAL[0], CATEGORICAL[1]],
        fliersize=2,
        linewidth=1,
    )
    ax.set_ylim(0, 1.02)
    ax.set_title("IoU / Dice vs. ground truth (anomalous images with masks)")
    style_axes(ax, grid_axis="y")
    save(fig, path)


def chart_defect_size_vs_score(stats_df: pd.DataFrame, path: Path) -> None:
    anomalous = stats_df[stats_df["GT_Label"] == 1].dropna(subset=["GT_Anomalous_Pixel_Pct"])
    if anomalous.empty:
        empty_chart(path, "No ground-truth masks available for this split")
        return
    fig, ax = new_fig((6.5, 5))
    caught = anomalous[anomalous["Classification"] == "TP"]
    missed = anomalous[anomalous["Classification"] == "FN"]
    if not caught.empty:
        ax.scatter(caught["GT_Anomalous_Pixel_Pct"], caught["Pred_Score"], color=STATUS["good"], label="Caught (TP)", alpha=0.75, s=28)
    if not missed.empty:
        ax.scatter(missed["GT_Anomalous_Pixel_Pct"], missed["Pred_Score"], color=STATUS["critical"], label="Missed (FN)", alpha=0.75, s=28)
    ax.set_xlabel("Ground-truth anomalous area (% of image)")
    ax.set_ylabel("Predicted anomaly score (raw)")
    ax.set_title("Defect size vs. detectability")
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=CHROME["ink_secondary"])
    save(fig, path)


def chart_pixel_confusion_matrix(counts: dict[str, int], path: Path) -> None:
    chart_confusion_matrix(counts, path, title="Pixel-level confusion matrix (aggregated)")
