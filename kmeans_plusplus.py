import numpy as np
from utils import get_cluster_assignments, get_all_dists_to_centers, get_min_dists_to_centers

def cluster_pp(points, k, weights, double_k=True):
    # kmeans++ with 2k gives an O(1) approximation while with just k it gives a log(k) approx
    if double_k:
        k *= 2

    n, d = int(points.shape[0]), int(points.shape[1])
    centers = [np.random.choice(n)]
    sq_dists = None
    while len(centers) < k:
        sq_dists = get_min_dists_to_centers(points, points[np.array(centers)[-1]], sq_dists)
        weighted_sq_dists = sq_dists * weights
        probs = weighted_sq_dists / np.sum(weighted_sq_dists)
        if np.isnan(probs).any():
            probs = np.ones_like(probs) / len(probs)
        centers.append(np.random.choice(n, p=probs))
    centers = np.array(centers)
    assignments, costs = get_cluster_assignments(points, centers, points[centers])
    return centers, assignments, costs
