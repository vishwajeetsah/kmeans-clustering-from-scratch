"""Clustering metrics implemented from scratch on numpy.

* ``silhouette_samples`` / ``silhouette_score`` - per-point and mean silhouette
  (Rousseeuw, 1987). Exact O(n^2) implementation; keep n in the low hundreds.
* ``adjusted_rand_index`` - ARI against ground-truth labels (Hubert & Arabie).
* ``elbow_inertias`` - inertia curve over a range of k.
* ``find_knee`` - simple "maximum distance from the chord" knee detector used
  for the elbow plot, operating on (k, inertia) points.
"""

import numpy as np


def _pairwise_dist(X):
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(np.maximum(sq, 0.0))


def silhouette_samples(X, labels):
    """Per-sample silhouette coefficient in [-1, 1]."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    D = _pairwise_dist(X)
    uniq = np.unique(labels)
    sils = np.empty(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        a = D[i][same].mean() if same.any() else 0.0
        b = np.inf
        for c in uniq:
            if c == labels[i]:
                continue
            other = labels == c
            if other.any():
                b = min(b, D[i][other].mean())
        sils[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return sils


def silhouette_score(X, labels):
    """Mean silhouette; 0.0 when there is only one cluster (undefined)."""
    X = np.asarray(X, dtype=float)
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(np.mean(silhouette_samples(X, labels)))


def adjusted_rand_index(labels_true, labels_pred):
    """ARI in [-1, 1]; 1.0 = identical partitions up to label permutation."""
    lt = np.asarray(labels_true)
    lp = np.asarray(labels_pred)
    classes = np.unique(lt)
    clusters = np.unique(lp)
    cont = np.zeros((len(classes), len(clusters)), dtype=float)
    cidx = {c: i for i, c in enumerate(classes)}
    kidx = {k: i for i, k in enumerate(clusters)}
    for t, p in zip(lt, lp):
        cont[cidx[t], kidx[p]] += 1
    n = len(lt)

    def comb2(x):
        return x * (x - 1) / 2

    sum_comb = comb2(cont).sum()
    sum_rows = comb2(cont.sum(axis=1)).sum()
    sum_cols = comb2(cont.sum(axis=0)).sum()
    expected = sum_rows * sum_cols / comb2(n)
    maximum = (sum_rows + sum_cols) / 2
    return float((sum_comb - expected) / (maximum - expected)) if maximum != expected else 0.0


def elbow_inertias(X, k_range, **kmeans_kwargs):
    """Fit k-means for each k in k_range; return (ks, inertias)."""
    from .kmeans import KMeans
    ks, inertias = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, **kmeans_kwargs).fit(X)
        ks.append(k)
        inertias.append(km.inertia_)
    return np.array(ks), np.array(inertias)


def find_knee(ks, values):
    """Elbow via discrete curvature: the k with the largest second difference.

    The second difference ``(v[i-1]-v[i]) - (v[i]-v[i+1])`` measures how
    sharply the inertia curve bends at each interior point; the maximum is
    the classic "elbow". This discrete-curvature criterion is more stable
    than the chord-distance heuristic on smooth curves.
    """
    ks = np.asarray(ks, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(ks) < 4:
        raise ValueError("need at least 4 points to find a knee")
    second_diff = np.diff(np.diff(values))
    return int(ks[1 + int(np.argmax(second_diff))])
