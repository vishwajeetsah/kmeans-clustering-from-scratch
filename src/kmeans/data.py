"""Seeded synthetic 2-D datasets for clustering experiments.

Everything here uses numpy's legacy ``RandomState`` with an explicit seed so
results are byte-for-byte reproducible across runs.
"""

import numpy as np


def _rs(seed):
    return np.random.RandomState(seed)


def make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42):
    """Isotropic Gaussian blobs.

    Returns (X, y) where y holds the true cluster label of each point.
    """
    rs = _rs(random_state)
    if isinstance(centers, int):
        centers = rs.uniform(-8, 8, size=(centers, 2))
    centers = np.asarray(centers, dtype=float)
    per = np.full(len(centers), n_samples // len(centers))
    per[: n_samples % len(centers)] += 1
    X = np.vstack(
        [c + rs.normal(scale=cluster_std, size=(p, 2)) for c, p in zip(centers, per)]
    )
    y = np.repeat(np.arange(len(centers)), per)
    perm = rs.permutation(n_samples)
    return X[perm], y[perm]


def make_anisotropic_blobs(n_samples=300, random_state=42):
    """Linearly stretched blobs; harder for vanilla k-means than round ones."""
    rs = _rs(random_state)
    X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0,
                      random_state=random_state)
    T = np.array([[0.6, -0.6], [-0.4, 1.0]])
    return X @ T, y


def make_moons(n_samples=300, noise=0.05, random_state=42):
    """Two interleaving half-circles; the classic non-convex k-means killer."""
    rs = _rs(random_state)
    n1, n2 = n_samples // 2, n_samples - n_samples // 2
    t1 = rs.uniform(0, np.pi, n1)
    t2 = rs.uniform(0, np.pi, n2)
    outer = np.c_[np.cos(t1), np.sin(t1)]
    inner = np.c_[1 - np.cos(t2), 1 - np.sin(t2) - 0.5]
    X = np.vstack([outer, inner]) + rs.normal(scale=noise, size=(n_samples, 2))
    y = np.repeat([0, 1], [n1, n2])
    perm = rs.permutation(n_samples)
    return X[perm], y[perm]


def make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42):
    """Two concentric circles; not separable by centroid distance at all."""
    rs = _rs(random_state)
    n1, n2 = n_samples // 2, n_samples - n_samples // 2
    theta = rs.uniform(0, 2 * np.pi, n_samples)
    r = np.concatenate([np.full(n1, 1.0), np.full(n2, factor)])
    X = np.c_[r * np.cos(theta), r * np.sin(theta)]
    X += rs.normal(scale=noise, size=(n_samples, 2))
    y = np.repeat([0, 1], [n1, n2])
    perm = rs.permutation(n_samples)
    return X[perm], y[perm]
