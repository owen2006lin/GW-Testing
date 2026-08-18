"""The central experiment: hold shared structure fixed, vary only the embedding.

At every point on this sweep the two representations are embeddings of the
*same* manifold, so the amount of shared intrinsic structure is exactly 100%
throughout and does not vary. The only thing that changes is how tightly the
second embedding winds in its ambient space.

If a calibrated similarity score were measuring shared structure, it would be
flat across this sweep. Anything that decays is reporting a property of the
embedding, not a property of what the two representations have in common.

The gap/spacing ratio is recorded at every point so the regime where the
graph-geodesic estimator stops being trustworthy is visible rather than hidden.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isoblind.calibration import calibrate_suite
from isoblind.geodesics import (euclidean_distances, geodesic_distances,
                                isometry_residual)
from isoblind.manifolds import make_isometric_pair, radial_gap_to_spacing_ratio
from isoblind.metrics import default_metric_suite

ROOT = Path(__file__).resolve().parents[1]
N = 1500
N_PERM = 200
SEEDS = [0, 1]
TURNS = [0.15, 0.4, 0.7, 1.0, 1.3, 1.6]


def main():
    t0 = time.time()
    rows = []
    for turns in TURNS:
        ratio = radial_gap_to_spacing_ratio(turns, N)
        print(f"\n### turns = {turns}   (sheet gap / sample spacing = {ratio:.2f})")
        for seed in SEEDS:
            data = make_isometric_pair(n=N, view_a="plane", view_b="swiss_roll",
                                       d_a=32, d_b=64, seed=seed,
                                       view_kwargs={"turns": turns})
            # diagnostics: is the intrinsic geometry still being recovered?
            geo = isometry_residual(geodesic_distances(data["Y"], k=10),
                                    data["D_true"])["pearson_r"]
            euc = isometry_residual(euclidean_distances(data["Y"]),
                                    data["D_true"])["pearson_r"]
            print(f"  seed {seed}: geodesic-vs-truth r={geo:.4f}   "
                  f"ambient-vs-truth r={euc:.4f}")

            res = calibrate_suite(default_metric_suite(knn_k=10),
                                  data["X"], data["Y"],
                                  n_perm=N_PERM, seed=seed)
            for r in res:
                d = r.as_row()
                d.update(turns=turns, seed=seed, gap_ratio=ratio,
                         geodesic_fidelity=geo, ambient_fidelity=euc)
                rows.append(d)
                print(f"      {d['metric']:<26} cal={d['calibrated']:6.3f}")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "curvature_sweep.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n{'=' * 84}")
    print("CALIBRATED SCORE vs EMBEDDING CURVATURE (shared structure fixed at 100%)")
    print("=" * 84)
    metrics = []
    for r in rows:
        if r["metric"] not in metrics:
            metrics.append(r["metric"])
    hdr = "  ".join(f"{t:>6.2f}" for t in TURNS)
    print(f"{'metric':<26} {'family':<10} {hdr}")
    for m in metrics:
        fam = next(r["family"] for r in rows if r["metric"] == m)
        cells = []
        for t in TURNS:
            v = np.mean([r["calibrated"] for r in rows
                         if r["metric"] == m and r["turns"] == t])
            cells.append(f"{v:6.3f}")
        print(f"{m:<26} {fam:<10} " + "  ".join(cells))
    print(f"\n{'gap/spacing':<26} {'':<10} " +
          "  ".join(f"{radial_gap_to_spacing_ratio(t, N):6.2f}" for t in TURNS))
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
