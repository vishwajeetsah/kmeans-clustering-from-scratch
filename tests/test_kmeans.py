"""Tests for the from-scratch KMeans implementation."""

import numpy as np
import pytest

from kmeans.data import make_blobs, make_moons
from kmeans.kmeans import KMeans


def test_fit_predict_shapes():
    X, _ = make_blobs(n_samples=90, centers=3, random_state=0)
    km = KMeans(n_clusters=3, n_init=5, random_state=0).fit(X)
    assert km.cluster_centers_.shape == (3, 2)
    assert km.labels_.shape == (90,)
    assert set(km.labels_) <= {0, 1, 2}
    assert km.inertia_ > 0
    assert km.n_iter_ >= 1
    preds = km.predict(X[:10])
    assert preds.shape == (10,)


def test_recovers_well_separated_blobs():
    # centres far apart: a correct run must beat a trivial baseline
    centres = np.array([[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    X, y = make_blobs(n_samples=150, centers=centres, cluster_std=0.5,
                      random_state=1)
    km = KMeans(n_clusters=3, n_init=10, random_state=1).fit(X)
    # inertia should be far below the inertia of the global-mean single cluster
    single = KMeans(n_clusters=1, n_init=1, random_state=1).fit(X)
    assert km.inertia_ < single.inertia_ / 20
    # centres land near the true centres (up to permutation)
    found = km.cluster_centers_[np.argsort(km.cluster_centers_[:, 0])]
    assert np.allclose(found, centres, atol=0.5)


def test_predict_raises_before_fit():
    with pytest.raises(RuntimeError):
        KMeans(n_clusters=2).predict(np.zeros((5, 2)))


def test_bad_inputs():
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)
    with pytest.raises(ValueError):
        KMeans(n_clusters=2, init="smart")
    X, _ = make_blobs(n_samples=4, centers=2, random_state=0)
    with pytest.raises(ValueError):
        KMeans(n_clusters=10).fit(X)


def test_kmeans_plus_plus_beats_random_on_blobs():
    # averaged over seeds, k-means++ should not lose to uniform init
    X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.2,
                      random_state=7)
    def mean_inertia(init):
        vals = [KMeans(n_clusters=4, init=init, n_init=1, random_state=s).fit(X).inertia_
                for s in range(10)]
        return float(np.mean(vals))
    assert mean_inertia("kmeans++") <= mean_inertia("random") * 1.05


def test_reproducible_with_seed():
    X, _ = make_blobs(n_samples=120, centers=3, random_state=3)
    a = KMeans(n_clusters=3, n_init=5, random_state=9).fit(X)
    b = KMeans(n_clusters=3, n_init=5, random_state=9).fit(X)
    assert np.allclose(a.cluster_centers_, b.cluster_centers_)
    assert np.array_equal(a.labels_, b.labels_)


def test_empty_cluster_does_not_crash():
    # k == n with duplicate points forces empty clusters during Lloyd
    X = np.vstack([np.zeros((5, 2)), np.ones((5, 2))])
    km = KMeans(n_clusters=10, n_init=2, random_state=0).fit(X)
    assert len(np.unique(km.labels_)) <= 10
    assert np.isfinite(km.inertia_)
