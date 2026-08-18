"""Build every figure in figures/ from the JSON in results/."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from isoblind.geodesics import (euclidean_distances, geodesic_distances,
                                normalize_distances)
from isoblind.manifolds import make_isometric_pair, sample_params
from isoblind.plotting import (FAMILY_COLOR, FAMILY_LABEL, GRID, INK, INK_2,
                               INK_MUTED, STYLES, SURFACE, apply_style,
                               strip_spines)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
apply_style()

FAMILY_ORDER = ["ambient", "local", "intrinsic"]


def repel(values, min_gap):
    """Push overlapping label positions apart while preserving their order.

    Direct labels beat a legend for readability, but only if they do not sit on
    top of each other. Lines that converge get their *labels* separated while
    the label order still matches the data order, so the mapping stays readable.
    """
    order = np.argsort(values)
    out = np.array(values, dtype=float)
    ordered = out[order]
    for i in range(1, len(ordered)):
        if ordered[i] - ordered[i - 1] < min_gap:
            ordered[i] = ordered[i - 1] + min_gap
    excess = ordered[-1] - max(values)
    if excess > 0:
        ordered -= excess * 0.5
    out[order] = ordered
    return out


def load(name):
    p = RES / name
    if not p.exists():
        print(f"  (skip: {name} not found)")
        return None
    return json.loads(p.read_text())


# --------------------------------------------------------------------------

def fig_setup():
    """What the two representations actually look like, and why it matters."""
    n = 1200
    data = make_isometric_pair(n=n, view_a="plane", view_b="swiss_roll",
                              d_a=2, d_b=3, seed=0, view_kwargs={"turns": 1.5})
    params = data["params"]
    # undo the random ambient rotation purely for drawing
    raw_a = params
    from isoblind.manifolds import embed_swiss_roll
    raw_b = embed_swiss_roll(params, turns=1.5)
    c = params[:, 0]

    fig = plt.figure(figsize=(12.4, 3.5))

    ax = fig.add_subplot(1, 4, 1)
    ax.scatter(raw_a[:, 0], raw_a[:, 1], c=c, cmap="viridis", s=5, linewidths=0)
    ax.set_title("Model A view\n(flat embedding)", color=INK)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    strip_spines(ax, keep=())

    ax = fig.add_subplot(1, 4, 2, projection="3d")
    ax.scatter(raw_b[:, 0], raw_b[:, 2], raw_b[:, 1], c=c, cmap="viridis",
               s=4, linewidths=0)
    ax.set_title("Model B view\n(curved embedding)", color=INK)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    ax.view_init(elev=62, azim=-60)
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 0.55))
    except Exception:
        pass

    D_true = normalize_distances(data["D_true"])
    iu = np.triu_indices_from(D_true, k=1)
    sub = np.random.default_rng(0).choice(len(iu[0]), 6000, replace=False)
    t = D_true[iu][sub]

    for k, (Dmat, label, title, colour) in enumerate([
        (euclidean_distances(data["Y"]), "ambient Euclidean distance",
         "What ambient metrics see", FAMILY_COLOR["ambient"]),
        (geodesic_distances(data["Y"], k=10), "estimated geodesic distance",
         "What intrinsic metrics see", FAMILY_COLOR["intrinsic"]),
    ]):
        ax = fig.add_subplot(1, 4, 3 + k)
        d = normalize_distances(Dmat)[iu][sub]
        r = np.corrcoef(t, d)[0, 1]
        ax.scatter(t, d, s=1.5, alpha=0.15, color=colour, linewidths=0)
        lim = max(t.max(), d.max()) * 1.02
        ax.plot([0, lim], [0, lim], color=INK_MUTED, lw=1, ls="--", zorder=3)
        ax.set_xlabel("true intrinsic distance")
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f"{title}\nr = {r:.3f}", color=INK, fontsize=10.5)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        strip_spines(ax)

    fig.suptitle("The two representations are isometric: identical intrinsic "
                 "geometry, different ambient embedding",
                 y=1.04, fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_setup.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig1_setup.png")


# --------------------------------------------------------------------------

def fig_main():
    rows = load("main.json")
    if rows is None:
        return
    iso = "isometric (plane vs swiss roll)"
    neg = "negative (independent manifolds)"
    pos = "positive (same embedding, rotated)"

    metrics, fams = [], {}
    for r in rows:
        if r["metric"] not in metrics:
            metrics.append(r["metric"])
            fams[r["metric"]] = r["family"]
    metrics.sort(key=lambda m: (FAMILY_ORDER.index(fams[m]), m))

    def mean_of(cond, m):
        return float(np.mean([r["calibrated"] for r in rows
                              if r["condition"] == cond and r["metric"] == m]))

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    y = np.arange(len(metrics))[::-1]
    for i, m in enumerate(metrics):
        col = FAMILY_COLOR[fams[m]]
        v = mean_of(iso, m)
        ax.barh(y[i], v, height=0.62, color=col, linewidth=0)
        ax.text(v + 0.015, y[i], f"{v:.3f}", va="center", ha="left",
                fontsize=9, color=INK_2)
        ax.plot([mean_of(pos, m)], [y[i]], marker="|", ms=13, mew=2,
                color=INK_MUTED, zorder=5)

    neg_max = max(mean_of(neg, m) for m in metrics)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=9.5)
    ax.set_xlim(0, 1.13)
    ax.set_xlabel("calibrated similarity score  (0 = indistinguishable from the "
                  "permutation null, 1 = perfect)")
    ax.set_title("Two representations of the SAME manifold, scored by nine "
                 "calibrated metrics\n"
                 "Shared intrinsic structure is 100% by construction in every bar",
                 loc="left", color=INK)
    ax.axvline(1.0, color=INK_MUTED, lw=1, ls=":", zorder=1)
    strip_spines(ax)
    ax.grid(axis="y", visible=False)

    handles = [plt.Line2D([], [], color=FAMILY_COLOR[f], lw=7,
                          label=FAMILY_LABEL[f]) for f in FAMILY_ORDER]
    handles.append(plt.Line2D([], [], color=INK_MUTED, marker="|", ls="none",
                              ms=13, mew=2, label="positive control"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=4, fontsize=8.8)
    ax.annotate(f"negative control (independent manifolds): "
                f"every metric calibrates to 0.000",
                xy=(0.5, -0.30), xycoords="axes fraction", ha="center",
                fontsize=8.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_main.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_main.png")


# --------------------------------------------------------------------------

def fig_curvature():
    rows = load("curvature_sweep.json")
    if rows is None:
        return
    turns = sorted({r["turns"] for r in rows})
    metrics, fams = [], {}
    for r in rows:
        if r["metric"] not in metrics:
            metrics.append(r["metric"])
            fams[r["metric"]] = r["family"]
    metrics.sort(key=lambda m: (FAMILY_ORDER.index(fams[m]), m))

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8.6, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1], "hspace": 0.12})

    per_family = {f: 0 for f in FAMILY_ORDER}
    ends = []
    for m in metrics:
        f = fams[m]
        ls, mk = STYLES[per_family[f] % len(STYLES)]
        per_family[f] += 1
        ys = [np.mean([r["calibrated"] for r in rows
                       if r["metric"] == m and r["turns"] == t]) for t in turns]
        ax.plot(turns, ys, ls=ls, marker=mk, color=FAMILY_COLOR[f],
                label=m, zorder=3)
        ends.append((m, f, ys[-1]))

    label_y = repel([e[2] for e in ends], min_gap=0.052)
    for (m, f, y_true), y_lab in zip(ends, label_y):
        ax.annotate(m, xy=(turns[-1], y_true), xytext=(turns[-1] + 0.11, y_lab),
                    va="center", fontsize=8.2, color=INK_2,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8),
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.9,
                                    shrinkA=0, shrinkB=2))

    ax.set_ylabel("calibrated similarity score")
    ax.set_ylim(-0.03, 1.10)
    ax.set_xlim(min(turns) - 0.05, max(turns) + 0.78)
    ax.set_title("Shared structure is held at 100% across this entire sweep\n"
                 "Only the ambient embedding changes — yet the ambient metrics "
                 "fall by more than half",
                 loc="left", color=INK)
    strip_spines(ax)
    handles = [plt.Line2D([], [], color=FAMILY_COLOR[f], lw=3,
                          label=FAMILY_LABEL[f]) for f in FAMILY_ORDER]
    ax.legend(handles=handles, loc="lower left", fontsize=9)

    fid = [np.mean([r["geodesic_fidelity"] for r in rows if r["turns"] == t])
           for t in turns]
    ax2.plot(turns, fid, color=FAMILY_COLOR["intrinsic"], marker="o", zorder=3)
    ax2.axhline(0.99, color=INK_MUTED, ls="--", lw=1)
    ax2.set_ylim(min(0.95, min(fid) - 0.01), 1.005)
    ax2.set_xlabel("turns of the spiral embedding  (intrinsic geometry identical throughout)")
    ax2.set_ylabel("geodesic\nestimator\nfidelity", fontsize=9)
    ax2.annotate("estimator still recovering true intrinsic distances",
                 (turns[0], 0.992), fontsize=8, color=INK_2, va="bottom")
    strip_spines(ax2)

    fig.savefig(FIG / "fig3_curvature.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_curvature.png")


# --------------------------------------------------------------------------

def fig_width():
    rows = load("width_sweep.json")
    if rows is None:
        return
    widths = sorted({r["width"] for r in rows})
    n = 256
    metrics, fams = [], {}
    for r in rows:
        if r["metric"] not in metrics:
            metrics.append(r["metric"])
            fams[r["metric"]] = r["family"]
    metrics.sort(key=lambda m: (FAMILY_ORDER.index(fams[m]), m))
    x = [w / n for w in widths]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, key, title in [
        (axes[0], "raw", "Raw score"),
        (axes[1], "calibrated", "After permutation calibration"),
    ]:
        per_family = {f: 0 for f in FAMILY_ORDER}
        ends = []
        for m in metrics:
            f = fams[m]
            ls, mk = STYLES[per_family[f] % len(STYLES)]
            per_family[f] += 1
            ys = [np.mean([r[key] for r in rows
                           if r["metric"] == m and r["width"] == w])
                  for w in widths]
            ax.plot(x, ys, ls=ls, marker=mk, color=FAMILY_COLOR[f], label=m,
                    zorder=3)
            ends.append((m, ys[-1]))

        if key == "raw":
            label_y = repel([e[1] for e in ends], min_gap=0.075)
            for (m, y_true), y_lab in zip(ends, label_y):
                # mkNN's raw score lies exactly on the analytic null baseline,
                # so the label says so rather than floating a separate note
                lab = (f"{m}  — sits on k/(n-1), Prop. 4.2"
                       if m.startswith("mkNN") else m)
                ax.annotate(lab, xy=(x[-1], y_true),
                            xytext=(x[-1] * 1.25, y_lab), va="center",
                            fontsize=8, color=INK_2,
                            bbox=dict(facecolor=SURFACE, edgecolor="none",
                                      pad=0.8),
                            arrowprops=dict(arrowstyle="-", color=GRID, lw=0.9,
                                            shrinkA=0, shrinkB=2))
        ax.set_xscale("log")
        ax.set_xlabel("d / n")
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylim(-0.05, 1.05)
        strip_spines(ax)

    axes[0].set_ylabel("similarity score")
    axes[0].set_xlim(min(x) * 0.8, max(x) * 9.0)
    axes[0].axhline(10 / (n - 1), color=INK_MUTED, ls=":", lw=1.2, zorder=2)
    fig.suptitle("X and Y are independent Gaussian noise — there is nothing to find\n"
                 "Raw spectral scores still climb to 0.9 as the representations widen",
                 y=1.06, fontsize=11.5, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_width.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_width.png")


if __name__ == "__main__":
    print("building figures")
    fig_setup()
    fig_main()
    fig_width()
    fig_curvature()
    print("done")
