"""Tests for synthetic datasets and hierarchical clustering."""

import numpy as np

from kmeans.data import (
    make_blobs,
    make_anisotropic_blobs,
    make_moons,
    make_circles,
)
from kmeans.hierarchical import AgglomerativeClustering
from kmeans.kmeans import KMeans
from kmeans.metrics import adjusted_rand_index


def test_generators_shapes_and_labels():
    for gen in (make_blobs, make_moons, make_circles):
        X, y = gen(n_samples=100, random_state=0)
        assert X.shape == (100, 2)
        assert y.shape == (100,)
        assert set(np.unique(y)) == {0, 1} or len(np.unique(y)) >= 2
    X, y = make_anisotropic_blobs(n_samples=99, random_state=0)
    assert X.shape == (99, 2) and y.shape == (99,)


def test_generators_reproducible():
    X1, y1 = make_moons(n_samples=50, random_state=3)
    X2, y2 = make_moons(n_samples=50, random_state=3)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_hierarchical_on_blobs():
    X, y = make_blobs(n_samples=60, centers=2, cluster_std=0.5,
                      random_state=0)
    labels = AgglomerativeClustering(n_clusters=2, linkage="complete").fit(X).labels_
    assert len(np.unique(labels)) == 2
    assert adjusted_rand_index(y, labels) > 0.9


def test_hierarchical_single_linkage_chains():
    # single linkage follows a chain: points at 0,1,2,3 with one far outlier
    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
    labels = AgglomerativeClustering(n_clusters=2, linkage="single").fit(X).labels_
    assert len(np.unique(labels)) == 2
    counts = sorted(np.bincount(labels))
    assert counts == [1, 4]


def test_hierarchical_beats_kmeans_on_circles():
    # concentric circles: centroid distance is useless, chaining wins
    X, y = make_circles(n_samples=80, noise=0.02, random_state=3)
    km_labels = KMeans(n_clusters=2, n_init=10, random_state=3).fit(X).labels_
    h_labels = AgglomerativeClustering(n_clusters=2, linkage="single").fit(X).labels_
    assert adjusted_rand_index(y, h_labels) > adjusted_rand_index(y, km_labels)


def test_kmeans_fails_on_moons():
    # documents the honest limitation: k-means cannot learn moon shapes
    X, y = make_moons(n_samples=120, random_state=0)
    pred = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X).labels_
    assert adjusted_rand_index(y, pred) < 0.6
