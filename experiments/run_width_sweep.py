"""Reproduction check: does the calibration implementation behave as advertised?

Groger et al. (Props 4.1, 4.2) show that under the null of no relationship,
spectral metrics have an expected score of O(d/n) while neighbourhood metrics
have O(k/n). A wider model therefore scores higher for free, and any raw
scaling curve confounds "more similar" with "more dimensions per sample".

Reproduced here on their synthetic protocol: X and Y are drawn independently
from N(0, I_d) at fixed n, with d swept so that d/n spans two orders of
magnitude. Nothing is shared, so the correct answer is zero everywhere.

What should appear:
  - raw spectral scores drifting upward with d/n
  - raw mkNN flat at k/(n-1), independent of d
  - all calibrated scores pinned near zero regardless of width

A note on why the representations must be full-rank Gaussians rather than
manifolds lifted into R^d: a random orthonormal lift preserves every pairwise
distance and inner product exactly, so every metric here is invariant to it
and the sweep would be flat by construction. The confounder is driven by the
*effective* dimension of the representation, not by the width of the box it
is written in. That distinction is easy to lose and worth stating.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isoblind.calibration import calibrate
from isoblind.metrics import (EuclideanRSA, LinearCKA, MutualKNN, Procrustes,
                              RBFCKA)

ROOT = Path(__file__).resolve().parents[1]
N = 256
WIDTHS = [8, 32, 128, 512, 1024, 2048]
N_PERM = 200
SEEDS = [0, 1, 2, 3, 4]
KNN_K = 10


def suite():
    return [LinearCKA(), RBFCKA(sigma_mult=1.0), EuclideanRSA(),
            Procrustes(), MutualKNN(k=KNN_K)]


def main():
    t0 = time.time()
    rows = []
    for d in WIDTHS:
        print(f"\n### d = {d}   d/n = {d / N:.2f}")
        for seed in SEEDS:
            rng = np.random.default_rng(1000 * seed + d)
            X = rng.standard_normal((N, d))
            Y = rng.standard_normal((N, d))
            for m in suite():
                r = calibrate(m, X, Y, n_perm=N_PERM, seed=seed)
                row = r.as_row()
                row.update(width=d, d_over_n=d / N, seed=seed)
                rows.append(row)
        last = rows[-len(suite()):]
        print("   " + "  ".join(f"{r['metric']}={r['raw']:.4f}" for r in last))

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "width_sweep.json", "w") as f:
        json.dump(rows, f, indent=2)

    names = []
    for r in rows:
        if r["metric"] not in names:
            names.append(r["metric"])

    print(f"\n{'=' * 92}")
    print(f"NULL DRIFT: X, Y ~ N(0, I_d) independent, n = {N}")
    print("=" * 92)
    for label, key in [("RAW score (spectral metrics should drift up with d/n)", "raw"),
                       ("CALIBRATED score (should stay at zero)", "calibrated")]:
        print(f"\n{label}")
        print(f"{'metric':<22}" + "".join(f"  d={d:<7}" for d in WIDTHS))
        for m in names:
            cells = "".join(
                f"{np.mean([r[key] for r in rows if r['metric'] == m and r['width'] == d]):9.4f}"
                for d in WIDTHS)
            print(f"{m:<22}" + cells)

    # Proposition 4.2 says the mkNN null baseline is exactly k/(n-1).
    pred = KNN_K / (N - 1)
    obs = np.mean([r["null_mean"] for r in rows if r["metric"].startswith("mkNN")])
    print(f"\nProp 4.2 check: mkNN null baseline")
    print(f"  predicted k/(n-1) = {pred:.6f}")
    print(f"  observed mean     = {obs:.6f}    rel. error {abs(obs - pred) / pred:.2%}")

    print(f"\nelapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
