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
from experiment_utils.get_data import get_dataset


def get_coreset(sensitivities, m, points, labels, weights=None):
    replace = False
    if m > len(points):
        replace = True
    rng = np.random.default_rng()
    coreset_inds = rng.choice(np.arange(len(sensitivities)), size=m, replace=replace, p=sensitivities)

    if weights is None:
        weights = np.ones_like(labels)
    q_points = points[coreset_inds]
    q_labels = labels[coreset_inds]
    new_weights = 1 / sensitivities[coreset_inds]
    # Want our coreset to be an unbiased estimator, so the sum of the new weights
    #   has to equal the sum of the old weights
    new_weights *= np.sum(weights) / np.sum(new_weights)
    q_weights = weights[coreset_inds] * new_weights
 
    return q_points, q_weights, q_labels

def make_rough_coreset(
    points,
    k,
    eps,
    norm,
    oversample=10,
    double_k=False,
    hst_count_from_norm=True
):
    centers, labels, costs = fast_cluster_pp(
        points,
        k,
        eps,
        norm=norm,
        double_k=double_k,
        hst_count_from_norm=hst_count_from_norm
    )
    sensitivities = bound_sensitivities(centers, labels, costs)
    m = get_coreset_size(k, eps, double_k, oversample)
    q_points, q_weights, q_labels = get_coreset(sensitivities, m, points, labels)
    return q_points, q_weights, q_labels

def get_coreset_size(k, eps, double_k, oversample):
    if double_k:
        m = int(oversample * k / (eps ** 2))
    else:
        m = int(oversample * k * np.log(k) / (eps ** 2))
    return m

# FIXME -- this method doesn't reaaaally need to exist
def make_true_coreset(
    points,
    weights,
    k,
    eps,
    norm,
    double_k=False,
    kmeans_alg=cluster_pp_slow,
    **kwargs
):
    # O(ndk) coreset time
    centers, labels, costs = kmeans_alg(points, k, weights, double_k)
    costs *= weights
    sensitivities = bound_sensitivities(centers, labels, costs)

    m = get_coreset_size(k, eps, double_k, oversample=1)
    r_points, r_weights, r_labels = get_coreset(sensitivities, m, points, labels, weights=weights)
    return r_points, r_weights, r_labels

def uniform_coreset(
    points,
    k,
    eps,
    norm,
    double_k=False,
    kmeans_alg=cluster_pp_slow,
    weights=None,
    **kwargs
):
    # Uniform coreset size should be the same as the other coreset sizes
    #   to show that it is super fast but terrible quality
    m = get_coreset_size(k, eps, double_k, oversample=1)
    n = len(points)
    q_points = points[np.random.choice(n, m)]
    q_weights = np.ones(m) * float(n) / m
    q_labels = np.ones(m)
    return q_points, q_weights, q_labels

def sensitivity_coreset(
    points,
    k,
    eps,
    norm,
    double_k=False,
    kmeans_alg=cluster_pp_slow,
    weights=None,
    **kwargs
):
    if weights is None:
        weights = np.ones(len(points))
    q_points, q_weights, q_labels = make_true_coreset(
        points=points,
        weights=weights,
        k=k,
        eps=eps,
        norm=norm,
        double_k=double_k,
        kmeans_alg=kmeans_alg
    )
    return q_points, q_weights, q_labels

def fast_coreset(
    points,
    k,
    eps,
    norm,
    oversample=10,
    double_k=False,
    make_second_coreset=False,
    hst_count_from_norm=True,
    kmeans_alg=cluster_pp_slow,
    **kwargs
):
    q_points, q_weights, q_labels = make_rough_coreset(
        points,
        k,
        eps,
        norm,
        oversample,
        double_k,
        hst_count_from_norm
    )
    if make_second_coreset:
        q_points, q_weights, q_labels = make_true_coreset(
            points=q_points,
            weights=q_weights,
            k=k,
            eps=eps,
            norm=norm,
            double_k=double_k,
            kmeans_alg=kmeans_alg
        )
    return q_points, q_weights, q_labels

def evaluate_coreset(points, k, coreset, weights):
    # Cost of solution on coreset with respect to original dataset
    centers, _, _ = cluster_pp(coreset, k, weights=weights)
    coreset_assignments, coreset_costs = get_cluster_assignments(coreset, centers, coreset[centers])
    coreset_costs *= weights
    coreset_cost = np.sum(coreset_costs)

    dataset_assignments, dataset_costs = get_cluster_assignments(points, centers, coreset[centers])
    dataset_cost = np.sum(dataset_costs)
    acc = max(dataset_cost / coreset_cost, coreset_cost / dataset_cost)

    return acc

if __name__ == '__main__':
    g_norm = 2
    g_points, _ = get_dataset('blobs', 10000, 50)
    g_k = 100
    g_eps = 0.5
    g_alpha = 1
    g_oversample = 1
    g_points = jl_proj(g_points, g_k, g_eps)
    g_kmeans_alg = cluster_pp#_slow

    start = time()
    q_points, q_weights, _ = fast_coreset(
        g_points,
        g_k,
        g_eps,
        g_norm,
        oversample=g_oversample,
        kmeans_alg=g_kmeans_alg
    )

    g_weights = np.ones((len(g_points)))
    # q_points, q_weights, q_labels = sensitivity_coreset(
    #     g_points,
    #     g_k,
    #     g_eps,
    #     g_norm,
    #     kmeans_alg=g_kmeans_alg,
    #     weights=g_weights
    # )

    # q_points, q_weights, q_labels = uniform_coreset(g_points, g_k, g_eps, g_norm)
    end = time()
    print(end - start)

    start = time()
    # r_points, r_weights, r_labels = make_true_coreset(
    #     g_points,
    #     weights,
    #     g_k,
    #     g_eps,
    #     g_norm,
    #     g_alpha,
    #     kmeans_alg=g_kmeans_alg
    # )
    end = time()
    # print(end - start)

    print('Coreset cost ratio:', evaluate_coreset(g_points, g_k, q_points, q_weights))

    # Visualize
    # embedding = PCA(n_components=2).fit_transform(q_points)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=q_labels)
    # plt.show()
