"""Representational similarity metrics, organised by what geometry they can see.

Every metric here exposes the same two-stage interface:

    m = LinearCKA(); m.prepare(X, Y); m.score(perm)

`prepare` does the expensive per-representation work once. `score` evaluates
the metric under a row-permutation of Y and must be cheap, because the
permutation null in `calibration.py` calls it hundreds of times. Getting this
factorisation right is what makes calibrating a quadratic-cost metric on a
laptop feasible.

Three families:

  ambient / extrinsic   - LinearCKA, RBFCKA, EuclideanRSA, Procrustes
        Functions of ambient inner products or ambient distances. Invariant
        to rotation and isotropic scaling of each space, and to nothing else.

  local / topological   - MutualKNN
        Depends only on which points are near which, not how near.

  intrinsic / geometric - GeodesicRSA, GWMatching
        Functions of estimated geodesic distances, hence invariant to any
        isometry of the manifold into its ambient space.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

from .geodesics import euclidean_distances, geodesic_distances, normalize_distances


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _double_center(K: np.ndarray) -> np.ndarray:
    """H K H with H = I - 11^T/n."""
    m = K.mean(0, keepdims=True)
    return K - m - m.T + K.mean()


def _apply_perm(M: np.ndarray, perm: np.ndarray | None) -> np.ndarray:
    if perm is None:
        return M
    return M[np.ix_(perm, perm)]


def _upper(M: np.ndarray) -> np.ndarray:
    return M[np.triu_indices_from(M, k=1)]


class Metric:
    """Base class. Subclasses set `name`, `family`, `s_max`."""

    name = "metric"
    family = "ambient"
    s_max = 1.0

    def prepare(self, X: np.ndarray, Y: np.ndarray) -> "Metric":
        raise NotImplementedError

    def score(self, perm: np.ndarray | None = None) -> float:
        raise NotImplementedError


# --------------------------------------------------------------------------
# ambient / extrinsic family
# --------------------------------------------------------------------------

class LinearCKA(Metric):
    """Linear centered kernel alignment (Kornblith et al., 2019)."""

    name = "CKA (linear)"
    family = "ambient"

    def prepare(self, X, Y):
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        self.Kx = _double_center(Xc @ Xc.T)
        self.Ky = _double_center(Yc @ Yc.T)
        self.nx = np.linalg.norm(self.Kx)
        self.ny = np.linalg.norm(self.Ky)
        return self

    def score(self, perm=None):
        Ky = _apply_perm(self.Ky, perm)
        return float((self.Kx * Ky).sum() / (self.nx * self.ny + 1e-12))


class RBFCKA(Metric):
    """Kernel CKA with an RBF kernel at a multiple of the median distance.

    The bandwidth multiplier interpolates between local and global sensitivity:
    small values only let near-neighbours contribute, large values approach
    the linear/global regime.
    """

    family = "ambient"

    def __init__(self, sigma_mult: float = 1.0):
        self.sigma_mult = sigma_mult
        self.name = f"CKA (RBF, s={sigma_mult:g})"

    @staticmethod
    def _rbf(X, mult):
        D = euclidean_distances(X)
        sigma = mult * np.median(_upper(D))
        return np.exp(-(D**2) / (2 * sigma**2 + 1e-12))

    def prepare(self, X, Y):
        self.Kx = _double_center(self._rbf(X, self.sigma_mult))
        self.Ky = _double_center(self._rbf(Y, self.sigma_mult))
        self.nx = np.linalg.norm(self.Kx)
        self.ny = np.linalg.norm(self.Ky)
        return self

    def score(self, perm=None):
        Ky = _apply_perm(self.Ky, perm)
        return float((self.Kx * Ky).sum() / (self.nx * self.ny + 1e-12))


class EuclideanRSA(Metric):
    """Spearman correlation between ambient Euclidean distance matrices.

    Second-order isomorphism in the sense of Kriegeskorte et al. (2008), but
    computed on ambient distances -- so it inherits the extrinsic view.
    """

    name = "RSA (Euclidean)"
    family = "ambient"

    def prepare(self, X, Y):
        self.Dx = euclidean_distances(X)
        self.Dy = euclidean_distances(Y)
        self._rx = rankdata(_upper(self.Dx))
        self._rx = (self._rx - self._rx.mean()) / (self._rx.std() + 1e-12)
        self._iu = np.triu_indices_from(self.Dx, k=1)
        return self

    def score(self, perm=None):
        Dy = _apply_perm(self.Dy, perm)
        ry = rankdata(Dy[self._iu])
        ry = (ry - ry.mean()) / (ry.std() + 1e-12)
        return float((self._rx * ry).mean())


class Procrustes(Metric):
    """Orthogonal Procrustes similarity, ||X^T Y||_* / (||X||_F ||Y||_F).

    Equals 1 exactly when the two clouds are related by an orthogonal map and
    uniform scaling. Like CKA it is a function of ambient inner products.

    Implementation note: the naive form takes an SVD of a d_x by d_y matrix on
    every permutation draw, which dominates the whole calibration once d gets
    into the thousands. Writing X = U_x S_x V_x^T and Y = U_y S_y V_y^T with
    thin SVDs, the nonzero singular values of X^T Y are those of
    S_x (U_x^T U_y) S_y, whose size is min(n, d) rather than d. Permuting rows
    of Y just permutes rows of U_y, so the thin SVDs are computed once and each
    draw costs an SVD of an r by r matrix with r <= n.
    """

    name = "Procrustes"
    family = "ambient"

    def prepare(self, X, Y):
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        Xc = Xc / (np.linalg.norm(Xc) + 1e-12)
        Yc = Yc / (np.linalg.norm(Yc) + 1e-12)

        Ux, Sx, _ = np.linalg.svd(Xc, full_matrices=False)
        Uy, Sy, _ = np.linalg.svd(Yc, full_matrices=False)
        tol = 1e-10
        kx, ky = int((Sx > tol).sum()), int((Sy > tol).sum())
        self.Ux, self.Sx = Ux[:, :kx], Sx[:kx]
        self.Uy, self.Sy = Uy[:, :ky], Sy[:ky]
        return self

    def score(self, perm=None):
        Uy = self.Uy if perm is None else self.Uy[perm]
        M = (self.Sx[:, None] * (self.Ux.T @ Uy)) * self.Sy[None, :]
        return float(np.linalg.svd(M, compute_uv=False).sum())


# --------------------------------------------------------------------------
# local / topological family
# --------------------------------------------------------------------------

class MutualKNN(Metric):
    """Mutual k-nearest-neighbour overlap (Huh et al., 2024).

    Purely ordinal: it asks which points are neighbours, never how far apart
    they are. Groger et al. show its null baseline is O(k/n) rather than
    O(d/n), which is why it survives calibration where spectral metrics do not.
    """

    family = "local"

    def __init__(self, k: int = 10):
        self.k = k
        self.name = f"mkNN (k={k})"

    @staticmethod
    def _adj(X, k):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        idx = nn.kneighbors(X, return_distance=False)[:, 1:]
        A = np.zeros((len(X), len(X)), dtype=np.float32)
        A[np.arange(len(X))[:, None], idx] = 1.0
        return A

    def prepare(self, X, Y):
        self.Ax = self._adj(X, self.k)
        self.Ay = self._adj(Y, self.k)
        self.n = len(X)
        return self

    def score(self, perm=None):
        Ay = _apply_perm(self.Ay, perm)
        return float((self.Ax * Ay).sum() / (self.n * self.k))


# --------------------------------------------------------------------------
# intrinsic / geometric family
# --------------------------------------------------------------------------

class GeodesicRSA(Metric):
    """Spearman correlation between *geodesic* distance matrices.

    Identical in form to EuclideanRSA; the only change is that distances are
    measured along the manifold instead of through the ambient space. That one
    substitution is the entire difference between a metric that is blind to
    isometric re-embedding and one that is not.
    """

    family = "intrinsic"

    def __init__(self, k: int = 10):
        self.k = k
        self.name = f"RSA (geodesic, k={k})"

    def prepare(self, X, Y):
        self.Dx = normalize_distances(geodesic_distances(X, k=self.k))
        self.Dy = normalize_distances(geodesic_distances(Y, k=self.k))
        self._iu = np.triu_indices_from(self.Dx, k=1)
        rx = rankdata(self.Dx[self._iu])
        self._rx = (rx - rx.mean()) / (rx.std() + 1e-12)
        return self

    def score(self, perm=None):
        Dy = _apply_perm(self.Dy, perm)
        ry = rankdata(Dy[self._iu])
        ry = (ry - ry.mean()) / (ry.std() + 1e-12)
        return float((self._rx * ry).mean())


class GWMatching(Metric):
    """Top-1 correspondence accuracy of the Gromov-Wasserstein optimal coupling.

    Gromov-Wasserstein compares two spaces through their *internal* distance
    matrices only. It never places the two clouds in a common coordinate
    system, so it needs no shared embedding and no known correspondence -- it
    solves for the correspondence.

    That creates a wrinkle for permutation calibration. The raw GW discrepancy
    is invariant to relabelling the samples of either space, so permuting rows
    of Y leaves it exactly unchanged and the permutation null is degenerate.
    What *is* permutation-sensitive, and what actually matters for transfer
    between models, is whether the recovered coupling lands on the true
    correspondence. That is what this metric reports.

    Because the optimal coupling under a permuted Y is just the permuted
    optimal coupling, the entire null distribution follows from a single GW
    solve.
    """

    family = "intrinsic"

    def __init__(self, k: int = 10, max_iter: int = 200, use_geodesic: bool = True):
        self.k = k
        self.max_iter = max_iter
        self.use_geodesic = use_geodesic
        tag = "geodesic" if use_geodesic else "Euclidean"
        self.name = f"GW matching ({tag})"

    def prepare(self, X, Y):
        import ot

        if self.use_geodesic:
            Cx = geodesic_distances(X, k=self.k)
            Cy = geodesic_distances(Y, k=self.k)
        else:
            Cx, Cy = euclidean_distances(X), euclidean_distances(Y)
        Cx, Cy = normalize_distances(Cx), normalize_distances(Cy)

        n = len(X)
        p = np.ones(n) / n
        P, log = ot.gromov.gromov_wasserstein(
            Cx, Cy, p, p, loss_fun="square_loss",
            log=True, max_iter=self.max_iter, tol_rel=1e-9,
        )
        self.coupling = P
        self.gw_dist = float(log["gw_dist"])
        # argmax_a P[i, a] is the point of Y that GW believes matches X_i
        self.match = np.asarray(P).argmax(axis=1)
        self.n = n
        return self

    def score(self, perm=None):
        target = np.arange(self.n) if perm is None else perm
        return float((target == self.match).mean())


# --------------------------------------------------------------------------

def default_metric_suite(knn_k: int = 10) -> list[Metric]:
    """The panel used in the main experiment."""
    return [
        LinearCKA(),
        RBFCKA(sigma_mult=0.25),
        RBFCKA(sigma_mult=1.0),
        RBFCKA(sigma_mult=4.0),
        EuclideanRSA(),
        Procrustes(),
        MutualKNN(k=knn_k),
        GeodesicRSA(k=knn_k),
        GWMatching(k=knn_k),
    ]
