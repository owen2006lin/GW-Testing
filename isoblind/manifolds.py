"""Generators for point clouds whose *intrinsic* geometry is known exactly.

The central object is an isometric pair: two embeddings of the same abstract
Riemannian manifold into different ambient Euclidean spaces. By construction
the two point clouds have identical intrinsic (geodesic) distance matrices
while having very different extrinsic (ambient Euclidean) geometry.

The flat rectangle [0, S] x [0, H] is used as the abstract manifold throughout.
Because it is flat and convex, its geodesics are straight lines in the
(s, h) parameter chart and the intrinsic distance between two parameter
points is *exactly* their Euclidean distance in that chart. Any isometric
embedding must therefore preserve that distance matrix exactly, which gives
us an analytic ground truth to check numerical geodesic estimates against.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


# --------------------------------------------------------------------------
# Isometric embeddings of the flat rectangle
# --------------------------------------------------------------------------

def _spiral_arclength(t: np.ndarray | float) -> np.ndarray | float:
    """Arc length of the Archimedean spiral r(t) = (t cos t, t sin t) from 0 to t.

    |r'(t)| = sqrt(1 + t^2), so s(t) = [t sqrt(1+t^2) + asinh(t)] / 2.
    """
    return 0.5 * (t * np.sqrt(1.0 + t**2) + np.arcsinh(t))


def _invert_arclength(s: np.ndarray, t_max: float) -> np.ndarray:
    """Numerically invert s(t), returning the t with _spiral_arclength(t) = s."""
    out = np.empty_like(s, dtype=float)
    for i, si in enumerate(s):
        out[i] = brentq(lambda t: _spiral_arclength(t) - si, 0.0, t_max, xtol=1e-12)
    return out


def embed_plane(params: np.ndarray) -> np.ndarray:
    """The trivial isometric embedding: the chart itself, in R^2."""
    return params.copy()


def embed_swiss_roll(params: np.ndarray, turns: float = 1.1,
                     t_min: float = 1.5) -> np.ndarray:
    """Isometric embedding of the flat chart onto a spiral (Swiss roll) in R^3.

    This is a *generalized cylinder* over an Archimedean spiral r(t) = a(t cos t,
    t sin t). Such surfaces are developable, hence flat, hence isometric to a
    planar region.

    Two details decide whether the map is a true isometry or merely a
    diffeomorphism that looks like one:

      1. The first coordinate must be *arc length* along the spiral, not the
         spiral parameter t. The textbook `make_swiss_roll` generator uses t,
         which stretches the surface non-uniformly.
      2. The transverse coordinate h must carry the *same* scale as s. It is
         tempting to rescale s onto whatever arc-length range the spiral
         happens to span while leaving h alone, but that is an anisotropic
         scaling of the chart and it destroys the isometry.

    Here the spiral is instead scaled by a single factor `a` chosen so that its
    total arc length equals the chart's own s-range, leaving both chart
    coordinates untouched. The induced first fundamental form is then exactly
    ds^2 + dh^2.

    `turns` is the number of times the chart wraps around the origin. It is the
    experiment's independent variable: the intrinsic geometry is identical at
    every setting, while the ambient geometry becomes progressively less
    informative about it as the strip winds more tightly. See
    `radial_gap_to_spacing_ratio` for the sampling density this demands.
    """
    s, h = params[:, 0], params[:, 1]
    s_span = s.max() - s.min()
    t_max = t_min + 2.0 * np.pi * turns

    l_lo, l_hi = _spiral_arclength(t_min), _spiral_arclength(t_max)
    a = s_span / (l_hi - l_lo)                    # unit-arc-length calibration

    target = l_lo + (s - s.min()) / a
    t = _invert_arclength(target, t_max=t_max * 1.5)
    return np.column_stack([a * t * np.cos(t), h, a * t * np.sin(t)])


def radial_gap_to_spacing_ratio(turns: float, n: int, t_min: float = 1.5,
                                area: float = 0.7) -> float:
    """How well separated are adjacent sheets of the roll, in units of sample spacing?

    The graph-geodesic estimator works by assuming that a point's ambient
    nearest neighbours are also its neighbours *on the manifold*. That fails as
    soon as adjacent sheets of the roll come closer together than the typical
    distance between samples, because the k-NN graph then acquires edges that
    tunnel between sheets and the estimated geodesics short-circuit.

    Adjacent sheets of an Archimedean spiral scaled by `a` are 2*pi*a apart,
    and samples on a domain of the given area sit roughly sqrt(area/n) apart.
    The ratio of the two is the quantity that has to stay comfortably above 1.
    Falling below ~2 is where the estimator starts to degrade in practice.
    """
    t_max = t_min + 2.0 * np.pi * turns
    a = 1.0 / (_spiral_arclength(t_max) - _spiral_arclength(t_min))
    return float(2.0 * np.pi * a / np.sqrt(area / n))


def embed_cylinder(params: np.ndarray, wraps: float = 0.85) -> np.ndarray:
    """Isometric embedding of the flat chart onto a cylinder in R^3.

    The radius is chosen so the chart wraps around the cylinder `wraps` times,
    which controls how badly ambient Euclidean distance misreports intrinsic
    distance. At wraps << 1 the cylinder is nearly flat and there is nothing to
    detect. wraps must stay below 1: past a full turn the strip passes through
    itself, which makes the map an immersion rather than an embedding and lets
    the k-NN graph short-circuit between sheets that are not actually adjacent.
    The angular coordinate is arc length over radius, which is what makes the
    map an isometry.
    """
    if wraps >= 1.0:
        raise ValueError("wraps must be < 1 or the strip self-intersects")
    s, h = params[:, 0], params[:, 1]
    radius = (s.max() - s.min()) / (wraps * 2.0 * np.pi)
    theta = (s - s.min()) / radius
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), h])


EMBEDDINGS = {
    "plane": embed_plane,
    "swiss_roll": embed_swiss_roll,
    "cylinder": embed_cylinder,
}


# --------------------------------------------------------------------------
# Ambient nuisance transforms (the "different model" part)
# --------------------------------------------------------------------------

def lift_to_ambient(X: np.ndarray, d_ambient: int, rng: np.random.Generator,
                    scale: float = 1.0) -> np.ndarray:
    """Embed X into R^d_ambient via a random orthonormal frame.

    This is the operation two different models would differ by even if they
    had learned the same thing: an arbitrary choice of basis in a wider space.
    It is an isometry of the ambient space, so it changes no intrinsic
    geometry at all, but it is exactly the kind of nuisance that inflates
    finite-sample similarity baselines as d_ambient grows.
    """
    d_in = X.shape[1]
    if d_ambient < d_in:
        raise ValueError(f"d_ambient={d_ambient} < intrinsic embedding dim {d_in}")
    A = rng.standard_normal((d_in, d_ambient))
    Q, _ = np.linalg.qr(A.T)            # (d_ambient, d_in) with orthonormal cols
    return scale * (X @ Q.T)


def add_noise(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Isotropic Gaussian noise, scaled relative to the cloud's own spread."""
    if sigma <= 0:
        return X
    spread = np.median(np.linalg.norm(X - X.mean(0), axis=1))
    return X + sigma * spread * rng.standard_normal(X.shape)


# --------------------------------------------------------------------------
# Top-level pair construction
# --------------------------------------------------------------------------

def sample_params(n: int, rng: np.random.Generator, aspect: float = 1.0,
                  jitter: bool = True, domain: str = "trapezoid") -> np.ndarray:
    """Sample n points from a flat, convex planar domain.

    A jittered grid is used rather than pure uniform sampling: it keeps the
    k-NN graph well connected, which matters for the geodesic estimator.

    The default domain is a scalene trapezoid rather than a rectangle, and the
    reason is worth stating. Gromov-Wasserstein recovers a correspondence only
    up to the self-isometries of the space it is given. A rectangle has a
    four-element symmetry group, so GW is entitled to return a flipped
    matching that is geometrically perfect but scores zero on top-1 accuracy.
    A domain with trivial symmetry group removes that ambiguity, and makes
    matching accuracy an honest readout of geometric recovery. Pass
    domain="rect" to see the degenerate case.
    """
    side = int(np.ceil(np.sqrt(n * 1.9)))
    gs, gh = np.meshgrid(np.linspace(0, 1, side),
                         np.linspace(0, aspect, side), indexing="ij")
    pts = np.column_stack([gs.ravel(), gh.ravel()])
    if jitter:
        pts = pts + rng.uniform(-0.5, 0.5, pts.shape) * np.array([1 / side, aspect / side])

    if domain == "trapezoid":
        # width tapers linearly with s, and the taper is off-centre
        keep = pts[:, 1] <= aspect * (0.95 - 0.55 * pts[:, 0]) + 0.02 * aspect
        pts = pts[keep]
    elif domain == "rect":
        pass
    else:
        raise ValueError(f"unknown domain {domain!r}")

    if len(pts) < n:
        raise ValueError(f"domain yielded {len(pts)} points, need {n}; raise the grid size")
    idx = rng.permutation(len(pts))[:n]
    return pts[idx]


def make_isometric_pair(n: int = 600, view_a: str = "plane",
                        view_b: str = "swiss_roll", d_a: int = 32, d_b: int = 32,
                        noise: float = 0.0, aspect: float = 1.0,
                        domain: str = "trapezoid", seed: int = 0,
                        view_kwargs: dict | None = None) -> dict:
    """Two ambient point clouds that are isometric by construction.

    Returns a dict with:
        X, Y      : (n, d_a), (n, d_b) ambient coordinates, rows in correspondence
        params    : (n, 2) shared chart coordinates
        D_true    : (n, n) exact intrinsic distance matrix, valid for both views
    """
    rng = np.random.default_rng(seed)
    params = sample_params(n, rng, aspect=aspect, domain=domain)

    vk = view_kwargs or {}
    raw_a = EMBEDDINGS[view_a](params)
    raw_b = EMBEDDINGS[view_b](params, **vk)

    # Exact intrinsic distances: the chart is flat and convex, so geodesic
    # distance in the abstract manifold is Euclidean distance in the chart.
    # Both embeddings are isometries, so this matrix describes both of them.
    diff = params[:, None, :] - params[None, :, :]
    D_true = np.sqrt((diff**2).sum(-1))

    X = lift_to_ambient(raw_a, d_a, np.random.default_rng(seed + 101))
    Y = lift_to_ambient(raw_b, d_b, np.random.default_rng(seed + 202))
    X = add_noise(X, noise, np.random.default_rng(seed + 303))
    Y = add_noise(Y, noise, np.random.default_rng(seed + 404))

    return {"X": X, "Y": Y, "params": params, "D_true": D_true,
            "view_a": view_a, "view_b": view_b}


def make_positive_control(n: int = 600, d_a: int = 32, d_b: int = 32,
                          view: str = "swiss_roll", noise: float = 0.0,
                          seed: int = 0) -> dict:
    """Same embedding on both sides, differing only by ambient rotation.

    Every metric considered here is invariant to ambient rotation, so all of
    them should score near their maximum. This is the check that the metric
    implementations are not simply broken.
    """
    return make_isometric_pair(n=n, view_a=view, view_b=view, d_a=d_a, d_b=d_b,
                               noise=noise, seed=seed)


def make_negative_control(n: int = 600, d_a: int = 32, d_b: int = 32,
                          seed: int = 0) -> dict:
    """Two independently generated manifolds with no shared structure.

    Calibrated scores should be at or near zero for every metric. This is the
    Type-I error check on the calibration procedure.
    """
    rng = np.random.default_rng(seed)
    pa = sample_params(n, rng)
    pb = sample_params(n, np.random.default_rng(seed + 7777))

    X = lift_to_ambient(embed_swiss_roll(pa), d_a, np.random.default_rng(seed + 11))
    Y = lift_to_ambient(embed_swiss_roll(pb), d_b, np.random.default_rng(seed + 22))

    diff = pa[:, None, :] - pa[None, :, :]
    return {"X": X, "Y": Y, "params": pa,
            "D_true": np.sqrt((diff**2).sum(-1)),
            "view_a": "swiss_roll", "view_b": "swiss_roll (independent sample)"}
