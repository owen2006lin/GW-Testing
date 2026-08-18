"""Permutation null-calibration, following Groger, Wen & Brbic (2026).

The problem it solves: a raw similarity score has no meaningful zero. Under
the null of no relationship whatsoever, the expected score

    mu_0(n, d_x, d_y) = E_pi[ s(X, pi(Y)) ]

is not zero, and for spectral metrics it grows as O(d/n). So a wider model
scores higher for free, and a raw scaling curve confounds "more similar" with
"more dimensions per sample".

The fix is to stop reasoning about the score and start reasoning about where
it sits in its own null distribution. Permuting the row correspondence between
X and Y destroys any relationship while preserving both marginals, giving an
empirical null; the calibrated score measures how far above that null's upper
tail the observed score lies, rescaled so a perfect match still reads 1.

    tau_alpha  = (1 - alpha) quantile of the combined null + observed scores
    s_cal      = max( (s_obs - tau_alpha) / (s_max - tau_alpha), 0 )

The add-one p-value is super-uniform under the null, which gives finite-sample
Type-I control at level alpha.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalibrationResult:
    metric: str
    family: str
    raw: float
    calibrated: float
    p_value: float
    tau: float
    null_mean: float
    null_std: float
    significant: bool
    n_perm: int
    null_scores: np.ndarray = field(repr=False, default=None)

    def as_row(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "null_scores"}


def calibrate(metric, X: np.ndarray, Y: np.ndarray, n_perm: int = 200,
              alpha: float = 0.05, seed: int = 0,
              keep_null: bool = False) -> CalibrationResult:
    """Null-calibrate one metric on one pair of representations.

    Parameters
    ----------
    metric : Metric
        Any object exposing `prepare(X, Y)` and `score(perm)`.
    n_perm : int
        Number of null draws, K. The smallest attainable p-value is
        1 / (K + 1), so K = 200 supports alpha = 0.05 comfortably.
    """
    rng = np.random.default_rng(seed)
    metric.prepare(X, Y)

    s_obs = metric.score(None)
    n = len(X)
    null = np.array([metric.score(rng.permutation(n)) for _ in range(n_perm)])

    # Rank-based critical value over the combined (observed + null) multiset,
    # matching the paper's definition so that tau is a valid order statistic.
    combined = np.concatenate([null, [s_obs]])
    idx = int(np.ceil((1 - alpha) * (n_perm + 1))) - 1
    idx = min(max(idx, 0), len(combined) - 1)
    tau = float(np.sort(combined)[idx])

    p = (1 + int((null >= s_obs).sum())) / (n_perm + 1)

    s_max = getattr(metric, "s_max", 1.0)
    denom = s_max - tau
    s_cal = max((s_obs - tau) / denom, 0.0) if denom > 1e-12 else 0.0

    return CalibrationResult(
        metric=metric.name,
        family=getattr(metric, "family", "unknown"),
        raw=float(s_obs),
        calibrated=float(s_cal),
        p_value=float(p),
        tau=tau,
        null_mean=float(null.mean()),
        null_std=float(null.std()),
        significant=bool(p <= alpha),
        n_perm=n_perm,
        null_scores=null if keep_null else None,
    )


def calibrate_suite(metrics: list, X: np.ndarray, Y: np.ndarray,
                    n_perm: int = 200, alpha: float = 0.05, seed: int = 0,
                    verbose: bool = False) -> list[CalibrationResult]:
    out = []
    for m in metrics:
        if verbose:
            print(f"    {m.name:<28}", end="", flush=True)
        r = calibrate(m, X, Y, n_perm=n_perm, alpha=alpha, seed=seed)
        if verbose:
            print(f" raw={r.raw:6.3f}  cal={r.calibrated:6.3f}  p={r.p_value:.4f}")
        out.append(r)
    return out


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH step-up FDR control, for when many model pairs are tested at once."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    out = np.zeros(m, dtype=bool)
    if passed.any():
        cutoff = np.max(np.nonzero(passed)[0])
        out[order[: cutoff + 1]] = True
    return out
