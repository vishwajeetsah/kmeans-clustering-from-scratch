"""K-means clustering implemented from scratch on numpy.

* ``kmeans++`` (D^2) initialisation plus plain uniform initialisation
* Lloyd's iterative refinement with a relative-inertia convergence check
* Multiple random restarts, keeping the best inertia
* ``fit`` / ``predict`` / ``fit_predict`` API compatible with the
  scikit-learn calling convention
"""

import numpy as np


def _pairwise_sq(X, C):
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


class KMeans:
    def __init__(self, n_clusters=3, init="kmeans++", n_init=10,
                 max_iter=300, tol=1e-4, random_state=42):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if init not in ("kmeans++", "random"):
            raise ValueError("init must be 'kmeans++' or 'random'")
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    # -- initialisation -------------------------------------------------
    def _init_centers(self, X, rs):
        n, _ = X.shape
        k = self.n_clusters
        if self.init == "random":
            return X[rs.choice(n, k, replace=False)].copy()
        # k-means++: first centre uniform, each next one with prob ~ D^2
        centers = np.empty((k, X.shape[1]))
        centers[0] = X[rs.randint(n)]
        d2 = _pairwise_sq(X, centers[[0]])[:, 0]
        for i in range(1, k):
            total = d2.sum()
            if total == 0:  # all points identical so far
                centers[i:] = X[rs.choice(n, k - i, replace=True)]
                break
            probs = d2 / total
            centers[i] = X[rs.choice(n, p=probs)]
            new_d2 = _pairwise_sq(X, centers[[i]])[:, 0]
            d2 = np.minimum(d2, new_d2)
        return centers

    # -- single Lloyd run ------------------------------------------------
    def _lloyd(self, X, centers):
        n = X.shape[0]
        for it in range(self.max_iter):
            labels = _pairwise_sq(X, centers).argmin(axis=1)
            new_centers = centers.copy()
            for j in range(self.n_clusters):
                members = X[labels == j]
                if len(members):            # empty cluster: keep old centre
                    new_centers[j] = members.mean(axis=0)
            shift = np.abs(new_centers - centers).sum()
            centers = new_centers
            if shift <= self.tol:
                return centers, labels, it + 1
        labels = _pairwise_sq(X, centers).argmin(axis=1)
        return centers, labels, self.max_iter

    @staticmethod
    def _inertia(X, centers, labels):
        return _pairwise_sq(X, centers)[np.arange(len(X)), labels].sum()

    # -- public API -------------------------------------------------------
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < self.n_clusters:
            raise ValueError("need a 2-D array with at least n_clusters rows")
        base = np.random.RandomState(self.random_state)
        best = (np.inf, None, None, 0)
        for _ in range(max(self.n_init, 1)):
            rs = np.random.RandomState(base.randint(0, 2 ** 31 - 1))
            centers, labels, n_iter = self._lloyd(X, self._init_centers(X, rs))
            inertia = self._inertia(X, centers, labels)
            if inertia < best[0]:
                best = (inertia, centers, labels, n_iter)
        self.inertia_, self.cluster_centers_, self.labels_, self.n_iter_ = best
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("call fit() before predict()")
        X = np.asarray(X, dtype=float)
        return _pairwise_sq(X, self.cluster_centers_).argmin(axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_
