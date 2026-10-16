"""Agglomerative hierarchical clustering, implemented from scratch.

Deliberately simple O(n^3) implementation: fine for demos on a few hundred
points, and a useful contrast with k-means for non-convex shapes. Supports
single and complete linkage with a ``fit`` / ``fit_predict`` API.
"""

import numpy as np


class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage="complete"):
        if linkage not in ("single", "complete"):
            raise ValueError("linkage must be 'single' or 'complete'")
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        if n < self.n_clusters:
            raise ValueError("fewer points than clusters")
        sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
        D = np.sqrt(np.maximum(sq, 0.0))
        clusters = [{i} for i in range(n)]
        while len(clusters) > self.n_clusters:
            # find closest pair under the linkage rule
            best, bi, bj = np.inf, -1, -1
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    block = D[np.ix_(sorted(clusters[i]), sorted(clusters[j]))]
                    dist = block.min() if self.linkage == "single" else block.max()
                    if dist < best:
                        best, bi, bj = dist, i, j
            merged = clusters[bi] | clusters[bj]
            clusters = [c for t, c in enumerate(clusters) if t not in (bi, bj)]
            clusters.append(merged)
        labels = np.empty(n, dtype=int)
        for lab, c in enumerate(clusters):
            labels[sorted(c)] = lab
        self.labels_ = labels
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_
