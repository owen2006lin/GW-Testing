"""Type-I error check on the calibration procedure.

Everything in this repo hinges on being able to read a calibrated score of
zero as "no detectable shared structure". That reading is only legitimate if
the test actually controls its false-positive rate, so it is measured here
rather than assumed.

Under a true null the add-one permutation p-value is super-uniform, so over
many independent trials the fraction with p <= alpha should be at or slightly
below alpha. Two nulls are tested:

  gaussian   X, Y ~ N(0, I_d) independent -- the textbook null
  manifold   two independent samples from the same curved manifold, with no
             row correspondence between them. This is the harder and more
             realistic case: the two clouds have identical *shape* and differ
             only in which points were drawn and how they are ordered. A test
             that rejects here would be reporting shape similarity as though
             it were correspondence, which is exactly the confusion the
             permutation null is supposed to prevent.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isoblind.calibration import calibrate
from isoblind.manifolds import make_negative_control
from isoblind.metrics import (EuclideanRSA, GeodesicRSA, LinearCKA, MutualKNN,
                              Procrustes, RBFCKA)

ROOT = Path(__file__).resolve().parents[1]
N = 400
D = 64
N_PERM = 200
N_TRIALS = 60
ALPHA = 0.05


def suite():
    return [LinearCKA(), RBFCKA(sigma_mult=1.0), EuclideanRSA(),
            Procrustes(), MutualKNN(k=10), GeodesicRSA(k=10)]


def gaussian_null(trial):
    rng = np.random.default_rng(90000 + trial)
    return rng.standard_normal((N, D)), rng.standard_normal((N, D))


def manifold_null(trial):
    d = make_negative_control(n=N, d_a=D, d_b=D, seed=50000 + trial)
    return d["X"], d["Y"]


def main():
    t0 = time.time()
    rows = []
    for null_name, builder in [("gaussian", gaussian_null),
                               ("manifold", manifold_null)]:
        print(f"\n### null = {null_name}   ({N_TRIALS} trials)")
        for trial in range(N_TRIALS):
            X, Y = builder(trial)
            for m in suite():
                r = calibrate(m, X, Y, n_perm=N_PERM, seed=trial)
                row = r.as_row()
                row.update(null=null_name, trial=trial)
                rows.append(row)
            if (trial + 1) % 20 == 0:
                print(f"  {trial + 1}/{N_TRIALS}")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "type1.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n{'=' * 84}")
    print(f"TYPE-I ERROR at alpha = {ALPHA}   ({N_TRIALS} trials per cell)")
    print("=" * 84)
    names = []
    for r in rows:
        if r["metric"] not in names:
            names.append(r["metric"])

    print(f"{'metric':<24}{'gaussian':>22}{'manifold':>22}")
    print(f"{'':<24}{'rej.rate  mean cal':>22}{'rej.rate  mean cal':>22}")
    worst = 0.0
    for m in names:
        cells = ""
        for nl in ["gaussian", "manifold"]:
            sel = [r for r in rows if r["metric"] == m and r["null"] == nl]
            rate = float(np.mean([r["significant"] for r in sel]))
            cal = float(np.mean([r["calibrated"] for r in sel]))
            worst = max(worst, rate)
            cells += f"{rate:12.3f}{cal:10.4f}"
        print(f"{m:<24}{cells}")

    # 95% binomial interval for the observed rate under a true alpha=0.05
    se = np.sqrt(ALPHA * (1 - ALPHA) / N_TRIALS)
    print(f"\nExpected {ALPHA:.2f} +/- {1.96 * se:.3f} (binomial, {N_TRIALS} trials)")
    print(f"Worst observed rejection rate: {worst:.3f}")
    print("PASS" if worst <= ALPHA + 1.96 * se + 1e-9 else "REVIEW: rate above interval")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
