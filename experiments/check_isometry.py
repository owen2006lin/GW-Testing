"""Self-check: are the constructed embeddings actually isometric?

Everything in this repo rests on the claim that `plane` and `swiss_roll` are
two embeddings of the *same* Riemannian manifold. That claim is analytic, but
it is easy to get wrong in code (parameterise the spiral by t instead of arc
length and the isometry silently disappears), so it is checked numerically
here before any similarity metric is run.

Two checks:
  1. The graph-geodesic estimate on each ambient cloud should reproduce the
     analytic chart distance matrix.
  2. Ambient Euclidean distances should NOT reproduce it for the curved view.
     If they did, there would be nothing interesting to measure.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isoblind.geodesics import (euclidean_distances, geodesic_distances,
                                isometry_residual)
from isoblind.manifolds import make_isometric_pair


def main():
    print("=" * 74)
    print("ISOMETRY SELF-CHECK")
    print("=" * 74)

    ok = True
    for view_b in ["swiss_roll", "cylinder"]:
        data = make_isometric_pair(n=600, view_a="plane", view_b=view_b,
                                   d_a=32, d_b=32, seed=0)
        X, Y, D_true = data["X"], data["Y"], data["D_true"]

        gx = isometry_residual(geodesic_distances(X, k=10), D_true)
        gy = isometry_residual(geodesic_distances(Y, k=10), D_true)
        ex = isometry_residual(euclidean_distances(X), D_true)
        ey = isometry_residual(euclidean_distances(Y), D_true)

        print(f"\nplane  vs  {view_b}")
        print(f"  geodesic estimate vs analytic chart distances")
        print(f"    flat view   r = {gx['pearson_r']:.4f}   rel-RMSE = {gx['rel_rmse']:.4f}")
        print(f"    curved view r = {gy['pearson_r']:.4f}   rel-RMSE = {gy['rel_rmse']:.4f}")
        print(f"  ambient Euclidean vs analytic chart distances")
        print(f"    flat view   r = {ex['pearson_r']:.4f}   rel-RMSE = {ex['rel_rmse']:.4f}")
        print(f"    curved view r = {ey['pearson_r']:.4f}   rel-RMSE = {ey['rel_rmse']:.4f}")

        intrinsic_ok = gy["pearson_r"] > 0.97
        extrinsic_differs = ey["pearson_r"] < gy["pearson_r"] - 0.05
        print(f"  -> intrinsic geometry recovered: {intrinsic_ok}")
        print(f"  -> ambient geometry genuinely differs: {extrinsic_differs}")
        ok &= intrinsic_ok and extrinsic_differs

    print("\n" + "=" * 74)
    print("PASS" if ok else "FAIL")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
