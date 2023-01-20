import numpy as np
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

from bico_master.bico.core import BICO
from bico_master.bico.geometry.point import Point

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

def bico_coreset(points, k, m, allotted_time):
    bico = BICO(points.shape[1], 100, m)
    for row in tqdm(points, total=len(points)):
        bico.insert_point(Point(row))
    c = bico.get_coreset()
    q_weights = c[:, 0]
    q_points = c[:, 1:]
    return q_points, q_weights, np.ones_like(q_weights)

def uniform_coreset(points, m, **kwargs):
    # Uniform coreset size should be the same as the other coreset sizes
    #   to show that it is super fast but terrible quality
    n = len(points)
    q_points = points[np.random.choice(n, m)]
    q_weights = np.ones(m) * float(n) / m
    q_labels = np.ones(m)
    return q_points, q_weights, q_labels

def sensitivity_coreset(
    points,
    k,
    m,
    norm,
    kmeans_alg=cluster_pp_slow,
    allotted_time=np.inf,
    weights=None,
    **kwargs
):
    if weights is None:
        weights = np.ones(len(points))
    # O(ndk) coreset time
    centers, labels, costs = kmeans_alg(points, k, weights, allotted_time=allotted_time)
    costs *= weights
    sensitivities = bound_sensitivities(centers, labels, costs)

    r_points, r_weights, r_labels = get_coreset(sensitivities, m, points, labels, weights=weights)
    return r_points, r_weights, r_labels

def fast_coreset(
    points,
    k,
    m,
    norm,
    hst_count_from_norm=True,
    allotted_time=np.inf,
    **kwargs
):
    # FIXME -- I must be going crazy. Using fewer than k centers to do the clustering gives
    #          BETTER results on the gaussian mixture model dataset.
    #        - Using 1 center performs poorly. Using 10 centers performs better than 1.
    #          Then using 100 centers performs poorly again!
    # - This ALSO holds true in the standard sensitivity sampling case
    centers, labels, costs = fast_cluster_pp(
        points,
        k,
        norm=norm,
        hst_count_from_norm=hst_count_from_norm,
        allotted_time=allotted_time,
    )
    if centers is None:
        # Ran out of time building the trees so we need to randomly initialize the data
        print('Ran out of time while building trees! Making coreset uniformly at random.')
        assert labels is None and costs is None
        return uniform_coreset(points, m)

    sensitivities = bound_sensitivities(centers, labels, costs)
    q_points, q_weights, q_labels = get_coreset(sensitivities, m, points, labels)
    return q_points, q_weights, q_labels

def evaluate_coreset(points, k, coreset, weights):
    centers, _, _ = cluster_pp(coreset, k, weights=weights)

    coreset_assignments, coreset_costs = get_cluster_assignments(coreset, centers, coreset[centers])
    coreset_costs *= weights
    coreset_cost = np.sum(coreset_costs)

    dataset_assignments, dataset_costs = get_cluster_assignments(points, centers, coreset[centers])
    dataset_cost = np.sum(dataset_costs)
    if coreset_cost == 0:
        acc = np.inf
    else:
        acc = max(dataset_cost / coreset_cost, coreset_cost / dataset_cost)

    return acc

if __name__ == '__main__':
    g_norm = 1
    g_k = 40
    g_points, _ = get_dataset('kdd_cup', n_points=10000, D=50, k=g_k)
    g_m_scalar = 50
    g_allotted_time = np.inf
    g_hst_count_from_norm = True
    g_kmeans_alg = cluster_pp#_slow
    # g_points = jl_proj(g_points, g_k, eps=0.5)

    method = 'fast'
    # method = 'sensitivity'
    # method = 'bico'
    # method = 'uniform'

    start = time()
    g_weights = np.ones((len(g_points)))
    if method == 'fast':
        q_points, q_weights, q_labels = fast_coreset(
            g_points,
            g_k,
            g_k * g_m_scalar,
            g_norm,
            hst_count_from_norm=g_hst_count_from_norm,
            allotted_time=g_allotted_time
        )
    elif method == 'sensitivity':
        q_points, q_weights, q_labels = sensitivity_coreset(
            g_points,
            g_k,
            g_k * g_m_scalar,
            g_norm,
            kmeans_alg=g_kmeans_alg,
            weights=g_weights,
            allotted_time=g_allotted_time
        )
    elif method == 'bico':
        q_points, q_weights, q_labels = bico_coreset(g_points, g_k, g_k * g_m_scalar, g_allotted_time)
    else:
        q_points, q_weights, q_labels = uniform_coreset(g_points, g_k * g_m_scalar)
    end = time()
    print(end - start)
    print('Coreset cost ratio:', evaluate_coreset(g_points, g_k, q_points, q_weights))

    # Visualize
    # embedding = PCA(n_components=2).fit_transform(q_points)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=q_labels)
    # plt.show()
