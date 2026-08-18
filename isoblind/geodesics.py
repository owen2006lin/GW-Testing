"""Intrinsic distance estimation from ambient coordinates.

Nothing here is novel -- this is the Isomap graph-geodesic estimator -- but it
is the piece that lets a similarity metric see intrinsic rather than extrinsic
geometry, so it is worth isolating.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from sklearn.neighbors import kneighbors_graph


def geodesic_distances(X: np.ndarray, k: int = 10,
                       auto_connect: bool = True, max_k: int = 60) -> np.ndarray:
    """Estimate intrinsic distances via shortest paths on a k-NN graph.

    Locally, a manifold looks Euclidean, so short ambient hops approximate
    short geodesic hops. Chaining them recovers the global intrinsic metric
    even when the ambient embedding is heavily curved. The estimate degrades
    exactly where the manifold folds close to itself, since the graph then
    admits shortcuts that are not real geodesics.

    If the graph is disconnected, k is increased until it connects (this is
    the standard failure mode of Isomap on sparse samples).
    """
    n = X.shape[0]
    while True:
        G = kneighbors_graph(X, n_neighbors=k, mode="distance")
        G = G.maximum(G.T)                     # symmetrize
        n_comp, _ = connected_components(csr_matrix(G), directed=False)
        if n_comp == 1 or not auto_connect or k >= min(max_k, n - 1):
            break
        k = int(np.ceil(k * 1.5))

    D = shortest_path(csr_matrix(G), method="D", directed=False)
    if np.isinf(D).any():                      # still disconnected: cap
        finite_max = D[np.isfinite(D)].max()
        D[np.isinf(D)] = finite_max * 1.5
    return D


def euclidean_distances(X: np.ndarray) -> np.ndarray:
    """Plain ambient pairwise Euclidean distances."""
    sq = (X**2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    np.fill_diagonal(D2, 0.0)
    return np.sqrt(D2)


def normalize_distances(D: np.ndarray) -> np.ndarray:
    """Scale a distance matrix to unit mean.

    Two models can represent the same geometry at different overall scales.
    Since scale is not part of the intrinsic shape we are asking about, it is
    divided out before any comparison.
    """
    m = D[np.triu_indices_from(D, k=1)].mean()
    return D / (m + 1e-12)


def isometry_residual(D_est: np.ndarray, D_true: np.ndarray) -> dict:
    """How well does an estimated distance matrix match the analytic one?

    Used as a self-check that the constructed embeddings really are isometric
    and that the geodesic estimator is recovering the intrinsic metric.
    """
    iu = np.triu_indices_from(D_true, k=1)
    a, b = normalize_distances(D_est)[iu], normalize_distances(D_true)[iu]
    return {
        "pearson_r": float(np.corrcoef(a, b)[0, 1]),
        "rel_rmse": float(np.sqrt(((a - b) ** 2).mean()) / (b.mean() + 1e-12)),
        "max_abs_dev": float(np.abs(a - b).max()),
    }
