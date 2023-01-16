import numpy as np

def get_cluster_assignments(points, center_inds, center_pts):
    n, d, k = len(points), len(points[0]), len(center_inds)
    all_dists = np.zeros((n, k)).tolist()
    all_dists = get_all_dists_to_centers(all_dists, points, center_pts)

    # FIXME still technically numpy but it's not the meat of the for loops
    cluster_assignments = np.argmin(all_dists, axis=1)
    cluster_assignments = np.array(center_inds)[cluster_assignments]
    costs = np.min(all_dists, axis=1)
    return cluster_assignments, costs

def get_all_dists_to_centers(pc_dists, points, centers):
    for i, point in enumerate(points):
        for j, center in enumerate(centers):
            dist_to_point = sq_euc_dist(point, center)
            pc_dists[i][j] = dist_to_point
    return pc_dists

def sq_euc_dist(x, y):
    dist = [(x[i] - y[i]) ** 2 for i in range(len(x))]
    dist = sum(dist)
    return dist

# Not numpy
def get_min_dists_to_centers(points, new_center, min_dists):
    for i, point in enumerate(points):
        dist = sq_euc_dist(point, new_center)
        if dist < min_dists[i]:
            min_dists[i] = dist

def cluster_pp_slow(points, k, weights, double_k=True):
    points = points.tolist()
    weights = weights.tolist()
    # kmeans++ with 2k gives an O(1) approximation while with just k it gives a log(k) approx
    if double_k:
        k *= 2

    n, d = len(points), len(points[0])
    centers = [np.random.choice(n)]
    sq_dists = (np.ones((len(points))) * np.inf).tolist()
    while len(centers) < k:
        center_point = points[centers[-1]]
        get_min_dists_to_centers(points, center_point, sq_dists)
        weighted_sq_dists = [sq_d * weights[i] for i, sq_d in enumerate(sq_dists)]
        sum_errors = sum(weighted_sq_dists)
        probs = [w_sq_d / sum_errors for w_sq_d in weighted_sq_dists]
        centers.append(np.random.choice(n, p=probs))
    center_points = []
    for center in centers:
        center_points.append(points[center])
    assignments, costs = get_cluster_assignments(points, centers, center_points)
    centers = np.array(centers)
    points = np.array(points)
    return centers, assignments, costs

