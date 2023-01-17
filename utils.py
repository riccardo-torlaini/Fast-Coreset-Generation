import numba
import numpy as np
from sklearn.random_projection import SparseRandomProjection

def get_cost_per_center(centers, labels, costs):
    cost_per_center = np.zeros((len(centers)))
    for i in range(len(centers)):
        points_in_cluster = np.where(labels == centers[i])
        cost_per_center[i] = np.sum(costs[points_in_cluster])
    return cost_per_center

# FIXME -- does the alpha actually matter here?
def bound_sensitivities(centers, labels, costs, alpha=10):
    sensitivities = np.zeros((len(labels)))
    cost_per_center = get_cost_per_center(centers, labels, costs)
    for i in range(len(centers)):
        points_in_cluster = np.where(labels == centers[i])
        if cost_per_center[i] > 0:
            sensitivities[points_in_cluster] = costs[points_in_cluster] / cost_per_center[i]
        # FIXME -- what's a reasonable value for alpha?
        sensitivities[points_in_cluster] *= alpha
        if len(points_in_cluster[0]) > 0:
            sensitivities[points_in_cluster] += 1 / len(points_in_cluster[0])

    sensitivities /= np.sum(sensitivities)
    return sensitivities

def jl_proj(points, k, eps):
    jl_dim = np.ceil(np.log(k) / (eps ** 2)).astype(np.int32)
    jl_model = SparseRandomProjection(jl_dim)
    points = jl_model.fit_transform(points)
    return points

def get_cluster_assignments(points, center_inds, center_pts):
    n, d = int(points.shape[0]), int(points.shape[1])
    k = len(center_inds)
    all_dists = np.zeros((n, k))
    all_dists = get_all_dists_to_centers(all_dists, points, center_pts)
    cluster_assignments = np.argmin(all_dists, axis=1)
    cluster_assignments = center_inds[cluster_assignments]
    costs = np.min(all_dists, axis=1)
    return cluster_assignments, costs

def get_min_dists_to_centers(points, new_center, dists):
    if dists is None:
        dists = np.ones((len(points))) * np.inf
    if len(new_center.shape) == 1:
        new_center = np.expand_dims(new_center, axis=0)
    new_dists = np.sum((points - new_center) ** 2, axis=-1)
    improved_inds = new_dists < dists
    dists[improved_inds] = new_dists[improved_inds]
    return dists

def get_all_dists_to_centers(pc_dists, points, centers):
    for i, point in enumerate(points):
        dists_to_point = np.sum((np.expand_dims(point, 0) - centers) ** 2, axis=-1)
        pc_dists[i] = dists_to_point
    return pc_dists

@numba.njit(fastmath=True)
def tree_dist(diam, curr_depth, max_depth):
    return 4 * diam * (0.5 ** curr_depth - 0.5 ** max_depth)
