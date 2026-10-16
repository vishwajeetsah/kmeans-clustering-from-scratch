"""Tests for clustering metrics."""

import numpy as np

from kmeans.data import make_blobs
from kmeans.kmeans import KMeans
from kmeans.metrics import (
    silhouette_samples,
    silhouette_score,
    adjusted_rand_index,
    elbow_inertias,
    find_knee,
)


def test_silhouette_in_range_and_good_on_blobs():
    X, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.5,
                      random_state=0)
    km = KMeans(n_clusters=3, n_init=5, random_state=0).fit(X)
    s = silhouette_samples(X, km.labels_)
    assert s.shape == (120,)
    assert np.all(s >= -1.0) and np.all(s <= 1.0)
    assert silhouette_score(X, km.labels_) > 0.6


def test_silhouette_bad_for_wrong_k():
    X, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.5,
                      random_state=0)
    good = silhouette_score(X, KMeans(n_clusters=3, n_init=5,
                                      random_state=0).fit(X).labels_)
    bad = silhouette_score(X, KMeans(n_clusters=1, n_init=2,
                                     random_state=0).fit(X).labels_)
    assert good > bad


def test_ari_perfect_and_chance():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand_index(y, y) == 1.0
    # identical up to permutation is still perfect
    assert adjusted_rand_index(y, np.array([2, 2, 0, 0, 1, 1])) == 1.0
    # random pairing on large data is near zero
    rs = np.random.RandomState(0)
    a = rs.randint(0, 5, 1000)
    b = rs.randint(0, 5, 1000)
    assert abs(adjusted_rand_index(a, b)) < 0.02


def test_ari_kmeans_on_blobs_high():
    X, y = make_blobs(n_samples=150, centers=3, cluster_std=0.5,
                      random_state=2)
    pred = KMeans(n_clusters=3, n_init=10, random_state=2).fit(X).labels_
    assert adjusted_rand_index(y, pred) > 0.95


def test_elbow_monotone_and_knee_detected():
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.6,
                      random_state=5)
    # elbow search starts at k=2: the k=1 point is structurally inflated and
    # would dominate the chord geometry of the knee detector
    ks, inertias = elbow_inertias(X, range(2, 9), n_init=5, random_state=5)
    assert (np.diff(inertias) <= 0).all()  # inertia must not rise with k
    knee = find_knee(ks, inertias)
    assert knee == 3, f"expected knee at k=3, got {knee}"
