"""Main experiment: does null-calibration mistake isometry for absence of structure?

Three conditions, all with n=600 paired points and 200 null permutations:

  positive   identical embedding, different ambient basis
             -> every metric should fire; checks the implementations work
  isometric  same intrinsic manifold, different ambient embedding
             -> the case of interest
  negative   two independently sampled manifolds
             -> nothing should fire; checks Type-I error of the calibration

Results are written to results/main.json and results/main.csv.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isoblind.calibration import calibrate_suite
from isoblind.manifolds import (make_isometric_pair, make_negative_control,
                                make_positive_control)
from isoblind.metrics import default_metric_suite

ROOT = Path(__file__).resolve().parents[1]
N = 1500
N_PERM = 200
ALPHA = 0.05
SEEDS = [0, 1, 2]
TURNS = 1.5      # gap/spacing ratio 4.9 at n=1500; geodesic estimator valid
D_A, D_B = 32, 64


def run_condition(name, builder, seed):
    data = builder(seed)
    metrics = default_metric_suite(knn_k=10)
    res = calibrate_suite(metrics, data["X"], data["Y"],
                          n_perm=N_PERM, alpha=ALPHA, seed=seed)
    for r in res:
        r_d = r.as_row()
        r_d["condition"] = name
        r_d["seed"] = seed
        yield r_d


CONDITIONS = {
    "positive (same embedding, rotated)":
        lambda s: make_positive_control(n=N, d_a=D_A, d_b=D_B, seed=s),
    "isometric (plane vs swiss roll)":
        lambda s: make_isometric_pair(n=N, view_a="plane", view_b="swiss_roll",
                                      d_a=D_A, d_b=D_B, seed=s,
                                      view_kwargs={"turns": TURNS}),
    "isometric (plane vs cylinder)":
        lambda s: make_isometric_pair(n=N, view_a="plane", view_b="cylinder",
                                      d_a=D_A, d_b=D_B, seed=s),
    "negative (independent manifolds)":
        lambda s: make_negative_control(n=N, d_a=D_A, d_b=D_B, seed=s),
}


def main():
    t0 = time.time()
    rows = []
    for cname, builder in CONDITIONS.items():
        print(f"\n### {cname}")
        for seed in SEEDS:
            print(f"  seed {seed}")
            for row in run_condition(cname, builder, seed):
                rows.append(row)
                print(f"    {row['metric']:<26} raw={row['raw']:7.3f}  "
                      f"cal={row['calibrated']:6.3f}  p={row['p_value']:.4f}")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "main.json", "w") as f:
        json.dump(rows, f, indent=2)

    # aggregate across seeds
    import csv
    keys = sorted({(r["condition"], r["metric"], r["family"]) for r in rows})
    agg = []
    for cond, met, fam in keys:
        sel = [r for r in rows if r["condition"] == cond and r["metric"] == met]
        agg.append({
            "condition": cond, "metric": met, "family": fam,
            "raw_mean": float(np.mean([r["raw"] for r in sel])),
            "raw_std": float(np.std([r["raw"] for r in sel])),
            "cal_mean": float(np.mean([r["calibrated"] for r in sel])),
            "cal_std": float(np.std([r["calibrated"] for r in sel])),
            "p_max": float(np.max([r["p_value"] for r in sel])),
            "n_sig": int(sum(r["significant"] for r in sel)),
            "n_seeds": len(sel),
        })
    with open(ROOT / "results" / "main.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)

    print(f"\n{'=' * 78}")
    print("SUMMARY (mean calibrated score over seeds)")
    print("=" * 78)
    for cond in CONDITIONS:
        print(f"\n{cond}")
        for a in agg:
            if a["condition"] == cond:
                flag = "*" if a["n_sig"] == a["n_seeds"] else " "
                print(f"  {flag} {a['metric']:<26} [{a['family']:<9}] "
                      f"raw={a['raw_mean']:7.3f}  cal={a['cal_mean']:6.3f} "
                      f"+/-{a['cal_std']:.3f}")
    print(f"\n(* = significant at alpha={ALPHA} in all {len(SEEDS)} seeds)")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
