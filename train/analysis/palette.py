"""Shared color palette and matplotlib styling helpers for every chart in this package.

Color usage follows the data-viz skill's method: categorical hues assigned in a
fixed order (never by value), a single sequential hue for magnitude/heatmaps,
and the reserved status scale (good/critical) for anything that is really a
pass/fail encoding (normal vs anomalous, caught vs missed).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------------------------------------------------------
# Palette - reference instance from the data-viz skill (references/palette.md).
# Categorical hues are assigned in this fixed order only, never by value.
# -----------------------------------------------------------------------------
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQUENTIAL_BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CHROME = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "gridline": "#e1e0d9",
    "axis": "#c3c2b7",
}

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE_STEPS)

# Split display names / colors for the one-per-run split-composition chart.
SPLIT_ORDER = ["Train", "Validation", "Test", "Contamination"]


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor(CHROME["surface"])
    ax.figure.set_facecolor(CHROME["surface"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(CHROME["axis"])
    ax.tick_params(colors=CHROME["ink_secondary"])
    if grid_axis:
        ax.grid(axis=grid_axis, color=CHROME["gridline"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.title.set_color(CHROME["ink"])
    ax.xaxis.label.set_color(CHROME["ink_secondary"])
    ax.yaxis.label.set_color(CHROME["ink_secondary"])


def new_fig(figsize: tuple[float, float] = (7, 5)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(CHROME["surface"])
    return fig, ax


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def empty_chart(path: Path, message: str = "No data available") -> None:
    fig, ax = new_fig((6, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", color=CHROME["muted"], fontsize=12)
    ax.set_axis_off()
    save(fig, path)


def class_color_map(names: list[str]) -> dict[str, str]:
    """Stable categorical hue per class name (alphabetical, wraps past 8 slots).

    Alphabetical (not by current sort/rank) so a class keeps its color if the
    chart gets re-sorted - color follows the entity, never its rank.
    """
    ordered = sorted(set(names))
    return {name: CATEGORICAL[i % len(CATEGORICAL)] for i, name in enumerate(ordered)}


def relative_luminance(rgba: tuple[float, float, float, float]) -> float:
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
