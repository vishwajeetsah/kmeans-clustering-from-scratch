"""Command-line demo: elbow selection, silhouette, and honest limitations.

Compares k-means against ground truth on well-separated blobs (where it
shines) and on moons / circles (where it legitimately fails), then shows
agglomerative clustering on the same hard shapes.
"""

import argparse
import sys

import numpy as np

from kmeans.data import make_blobs, make_anisotropic_blobs, make_moons, make_circles
from kmeans.kmeans import KMeans
from kmeans.metrics import (
    silhouette_score,
    adjusted_rand_index,
    elbow_inertias,
    find_knee,
)
from kmeans.hierarchical import AgglomerativeClustering


def report(name, X, y_true, k=2, seed=42, hier_n=80):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    sil = silhouette_score(X, km.labels_)
    ari = adjusted_rand_index(y_true, km.labels_)
    print(f"\n{name} (k-means, k={k}, n={len(X)})")
    print(f"  inertia   : {km.inertia_:.2f}")
    print(f"  silhouette: {sil:.4f}   (higher = tighter, better separated)")
    print(f"  ARI vs truth: {ari:.4f}  (1.0 = perfect match)")
    # hierarchical on a small subsample (O(n^3) naive implementation)
    rs = np.random.RandomState(seed)
    idx = rs.choice(len(X), min(hier_n, len(X)), replace=False)
    hier = AgglomerativeClustering(n_clusters=k, linkage="single").fit(X[idx])
    hier_ari = adjusted_rand_index(y_true[idx], hier.labels_)
    print(f"  agglomerative(single, n={len(idx)}) ARI: {hier_ari:.4f}")
    return km


def main(argv=None):
    ap = argparse.ArgumentParser(description="k-means from-scratch demo")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    print("=== elbow method on well-separated blobs ===")
    Xb, yb = make_blobs(n_samples=300, centers=3, cluster_std=0.6,
                        random_state=args.seed)
    # elbow search starts at k=2: the k=1 inertia would dominate the knee
    # detector's chord geometry
    ks, inertias = elbow_inertias(Xb, range(2, 9), random_state=args.seed)
    for k, val in zip(ks, inertias):
        print(f"  k={k}: inertia={val:.1f}")
    knee = find_knee(ks, inertias)
    print(f"  -> knee detected at k={knee} (true k=3)")

    report("well-separated blobs", Xb, yb, k=3, seed=args.seed)
    Xa, ya = make_anisotropic_blobs(n_samples=300, random_state=args.seed)
    report("anisotropic blobs", Xa, ya, k=3, seed=args.seed)
    Xm, ym = make_moons(n_samples=300, random_state=args.seed)
    report("interleaving moons", Xm, ym, k=2, seed=args.seed)
    Xc, yc = make_circles(n_samples=300, random_state=args.seed)
    report("concentric circles", Xc, yc, k=2, seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
