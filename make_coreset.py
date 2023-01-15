import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

from kmeans_plusplus import cluster_pp
from kmeans_plusplus_slow import cluster_pp_slow
from fast_kmeans_plusplus import fast_cluster_pp
# from coreset_sandbox import HST, hst_dist, fit_tree, assert_hst_correctness
from hst import CubeHST, hst_dist
from multi_hst import make_multi_HST
from utils import bound_sensitivities, jl_proj, get_cluster_assignments

import sys
np.set_printoptions(threshold=np.inf)
sys.setrecursionlimit(5000)


def get_coreset(sensitivities, m, points, labels, weights=None):
    replace = False
    if m > len(points):
        replace = True
    rng = np.random.default_rng()
    coreset_inds = rng.choice(np.arange(len(sensitivities)), size=m, replace=replace, p=sensitivities)

    if weights is None:
        weights = np.ones_like(labels)
    points = points[coreset_inds]
    labels = labels[coreset_inds]
    # Want the sum of the weights to equal n
    weights = weights[coreset_inds] * (1 / sensitivities[coreset_inds]) / m
 
    return points, labels, weights

def make_rough_coreset(
    points,
    k,
    eps,
    norm,
    alpha,
    oversample=10,
    double_k=True,
):
    # FIXME -- do we need to do 2k here since we are doing (fast)kmeans++?
    # Alternatively, we only incur a log(k) distortion by doing it for k, which
    #   we can oversample our coreset by. That seems faster...
    # This should be one of the experiments
    centers, labels, costs = fast_cluster_pp(points, k, eps, norm=norm, double_k=double_k)
    sensitivities = bound_sensitivities(centers, labels, costs, alpha=alpha)

    if double_k:
        m = int(oversample * k / (eps ** 2))
    else:
        m = int(oversample * k * np.log(k) / (eps ** 2))
    q_points, q_labels, q_weights = get_coreset(sensitivities, m, points, labels)
    return q_points, q_weights, q_labels

def make_true_coreset(
    points,
    weights,
    k,
    eps,
    norm,
    alpha,
    double_k=True,
    kmeans_alg=cluster_pp
):
    # O(ndk) coreset time
    centers, labels, costs = kmeans_alg(points, k, weights, double_k)
    costs *= weights
    sensitivities = bound_sensitivities(centers, labels, costs, alpha=alpha)

    # Sampling the coreset based on the sensitivities
    if double_k:
        m = int(k / (eps ** 2))
    else:
        m = int(k * np.log(k) / (eps ** 2))
    r_points, r_labels, r_weights = get_coreset(sensitivities, m, points, labels, weights=weights)
    return r_points, r_weights, r_labels

def evaluate_coreset(points, k, coreset, weights):
    uniform_weights = np.ones((len(points)))
    centers, labels, costs = cluster_pp(points, k, weights=uniform_weights)
    total_cost = np.sum(costs)

    coreset_assignments, coreset_costs = get_cluster_assignments(coreset, centers, points[centers])
    coreset_costs *= weights
    coreset_cost = np.sum(coreset_costs)
    return max(total_cost / coreset_cost, coreset_cost / total_cost)

if __name__ == '__main__':
    n_points = 100000
    D = 1000
    num_centers = 1000
    g_alpha = 10
    g_norm = 2
    g_points, _ = make_blobs(n_points, D, centers=num_centers)
    g_k = 5000
    g_eps = 0.5
    g_points = jl_proj(g_points, g_k, g_eps)
    g_kmeans_alg = cluster_pp#_slow

    start = time()
    q_points, q_weights, _ = make_rough_coreset(g_points, g_k, g_eps, g_norm, g_alpha)
    # q_points, q_weights, q_labels = make_true_coreset(
    #     q_points,
    #     q_weights,
    #     g_k,
    #     g_eps,
    #     g_norm,
    #     g_alpha,
    #     kmeans_alg=g_kmeans_alg
    # )
    end = time()
    print(end - start)

    start = time()
    weights = np.ones((len(g_points)))
    r_points, r_weights, r_labels = make_true_coreset(
        g_points,
        weights,
        g_k,
        g_eps,
        g_norm,
        g_alpha,
        kmeans_alg=g_kmeans_alg
    )
    end = time()
    print(end - start)
    # print('Coreset cost ratio:', evaluate_coreset(g_points, g_k, q_points, q_weights))

    # Visualize
    # embedding = PCA(n_components=2).fit_transform(q_points)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=q_labels)
    # plt.show()
