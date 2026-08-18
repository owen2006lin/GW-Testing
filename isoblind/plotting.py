"""Shared plotting style.

Colour encodes the *family* a metric belongs to -- what geometry it can see --
rather than the individual metric. That is the distinction the figures are
about, and it keeps the categorical palette to three slots, which is the
largest set that clears colourblind-separation floors on an all-pairs
comparison. Individual metrics within a family are separated by line style and
marker, so identity never rests on colour alone.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e4e3de"

FAMILY_COLOR = {
    "ambient":   "#2a78d6",   # categorical slot 1
    "local":     "#eb6834",   # slot 2
    "intrinsic": "#1baf7a",   # slot 3
}

FAMILY_LABEL = {
    "ambient": "ambient / extrinsic",
    "local": "local / topological",
    "intrinsic": "intrinsic / geometric",
}

# within-family separation, so identity is never colour-alone
STYLES = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D"), ((0, (3, 1, 1, 1)), "v")]


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "figure.dpi": 140,
    })


def strip_spines(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
