# kmeans-clustering-from-scratch

Unsupervised clustering implemented on pure NumPy — no scikit-learn, no black
boxes. Lloyd's k-means with k-means++ initialisation, an elbow/knee selector,
from-scratch evaluation metrics (silhouette, adjusted Rand index), and a naive
agglomerative hierarchical clustering for contrast.

## What it does

- **K-means**: k-means++ (D²) and uniform initialisation, Lloyd's iteration
  with relative-shift convergence, multiple restarts keeping the best inertia,
  empty-cluster safe.
- **Model selection**: inertia curve + elbow detection by discrete curvature
  (largest second difference of inertia).
- **Metrics from scratch**: per-point/mean silhouette score, adjusted Rand
  index against ground truth.
- **Agglomerative hierarchical clustering**: single and complete linkage
  (deliberately simple O(n³) implementation — good for demos, not for big data).
- **Seeded synthetic datasets**: Gaussian blobs, anisotropic blobs,
  interleaving moons, concentric circles — including the shapes where k-means
  legitimately fails.

## Quickstart

```bash
pip install -r requirements.txt
python -m kmeans.demo            # prints elbow table + comparison report
pytest                          # 18 tests
```

Use it as a library:

```python
from kmeans.data import make_blobs
from kmeans.kmeans import KMeans
from kmeans.metrics import silhouette_score, elbow_inertias, find_knee

X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
ks, inertias = elbow_inertias(X, range(2, 9), random_state=42)
k = find_knee(ks, inertias)            # 3
labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
print(silhouette_score(X, labels))     # 0.8857
```

## How it works

K-means alternates two steps until the centres stop moving:

1. **Assignment** — every point joins its nearest centre (Lloyd's step).
2. **Update** — each centre moves to the mean of its members.

k-means++ seeds the first centre uniformly, then samples each next centre with
probability proportional to its squared distance from the nearest existing
centre, so the initial centres spread across the data instead of clumping.

Silhouette for a point compares its mean distance to its own cluster (*a*)
against its mean distance to the nearest other cluster (*b*): `(b − a) /
max(a, b)`. Near 1 means well-clustered, near 0 means on a boundary, negative
means probably misassigned.

## Measured results (seed 42, n = 300)

| Dataset | k-means ARI | k-means silhouette | single-linkage ARI (n=80 subsample) |
|---|---|---|---|
| Well-separated blobs | 1.0000 | 0.8857 | 1.0000 |
| Anisotropic blobs | 1.0000 | 0.6680 | 1.0000 |
| Interleaving moons | 0.2152 | 0.4873 | 1.0000 |
| Concentric circles | 0.0111 | 0.3744 | −0.0019 |

Elbow on 3 blobs (inertia 3503.6 → 204.1 → 178.6 → 153.5 → …): knee detected at
k = 3, matching the true cluster count.

## Honest limitations

- **K-means assumes round clusters.** On interleaving moons it scores ARI 0.22,
  on concentric circles ARI 0.01 — it partitions by centroid distance, so
  non-convex shapes defeat it. Single-linkage hierarchical clustering handles
  the moons perfectly (ARI 1.0).
- **Single linkage is fragile to noise.** On the noisy circles subsample it
  peeled off one outlier as its own cluster (ARI ≈ 0). Outlier trimming or
  density-based methods (e.g. DBSCAN) fix this.
- **The elbow is a heuristic.** It works when the inertia curve has a clear
  bend; on smooth curves it can disagree with the true k. Silhouette analysis
  is a more robust alternative.
- **Hierarchical clustering is O(n³)** here — it is a teaching
  implementation, fine for hundreds of points, not for thousands.
- **Silhouette is O(n²)** for the same reason; keep evaluation sets small.
- All demo data is synthetic 2-D; nothing here is tuned for high-dimensional
  or real-world data.

## Project structure

```
src/kmeans/
  data.py          seeded dataset generators (blobs, moons, circles)
  kmeans.py        KMeans: k-means++ init, Lloyd's algorithm, restarts
  metrics.py       silhouette, ARI, inertia curve, knee detection
  hierarchical.py  agglomerative clustering (single/complete linkage)
  demo.py          CLI comparison report
tests/             18 pytest tests covering all modules
```

## What I Learned

- k-means++ is not a minor tweak: over 10 seeds on overlapping blobs it
  consistently beat uniform initialisation, because a single bad seed can
  strand two centres in one blob forever.
- The elbow method's "obvious" implementation (distance from the chord) is
  seed-sensitive on smooth curves; discrete curvature (largest second
  difference) was stable across every configuration I tried.
- Writing silhouette from scratch made its cost visceral: O(n²) pairwise
  distances means you feel every extra hundred points.
- The most useful thing in this project is the failure table: ARI 1.0 →
  0.01 across datasets is a better argument for "know your algorithm's
  assumptions" than any textbook paragraph.
- Single-linkage chaining is beautiful on clean moons and comically fragile
  on noisy circles — a reminder that every clustering method buys its wins
  by paying for them somewhere else.
