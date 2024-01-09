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

def get_coreset_with_centers(centers, sensitivities, m, points, labels, weights=None):
    """
    Sample coreset from sensitivity values but use k centers from approximate solution as the first 
    elements in the coreset
    """
    replace = False
    if m > len(points):
        replace = True
    rng = np.random.default_rng()
    k = len(centers)
    if m <= k:
        return points[centers[:m]], np.ones(m) / m, np.ones(m)
    sample_coreset_inds = rng.choice(
        np.arange(len(sensitivities)),
        size=m-k,
        replace=replace,
        p=sensitivities
    )

    ### Sample coreset
    full_coreset_inds = np.concatenate([centers, sample_coreset_inds], axis=0)
    q_points = points[full_coreset_inds]
    q_labels = labels[full_coreset_inds]

    ### Get weights for the coreset
    # Each center that we got will represent n/m points
    #   - This is a rough estimate and should be improved on in the future
    if weights is None:
        weights = np.ones_like(labels)
    new_weights = 1 / sensitivities[sample_coreset_inds]
    center_weights = np.ones(k) * np.mean(new_weights)
    new_weights = np.concatenate([center_weights, new_weights], axis=0)
    # Want our coreset to be an unbiased estimator, so the sum of the new weights
    #   has to equal the sum of the old weights
    new_weights *= np.sum(weights) / np.sum(new_weights)
    q_weights = weights[full_coreset_inds] * new_weights
 
    return q_points, q_weights, q_labels

def get_coreset(sensitivities, m, points, labels, weights=None):
    """
    Sample all m coreset elements from probability distribution values, ignoring the first k centers
    """
    replace = False
    if m > len(points):
        m = len(points)
        replace = True
    rng = np.random.default_rng()
    if m > len(sensitivities):
        m = len(sensitivities)
    coreset_inds = rng.choice(np.arange(len(sensitivities)), size=m, replace=False, p=sensitivities)

    if weights is None:
        weights = np.ones_like(labels)
    q_points = points[coreset_inds]
    q_labels = labels[coreset_inds]
    q_weights = 1 / sensitivities[coreset_inds]
    # q_weights = weights[coreset_inds] * new_weights
    # Want our coreset to be an unbiased estimator, so the sum of the new weights
    #   has to equal the sum of the old weights
    q_weights *= np.sum(weights) / np.sum(q_weights)
 
    return q_points, q_weights, q_labels

def bico_coreset(points, k, m, **kwargs):
    bico = BICO(points.shape[1], 100, m)
    for row in tqdm(points, total=len(points)):
        bico.insert_point(Point(row))
    c = bico.get_coreset()
    q_weights = c[:, 0]
    q_points = c[:, 1:]
    return q_points, q_weights, np.ones_like(q_weights)

def uniform_coreset(points, m, weights=None, **kwargs):
    n = len(points)
    if weights is None:
        weights = np.ones(n)
    if m > len(points):
        m = len(points)
    weights_prob_dist = weights / np.sum(weights)
    if m > n:
        m = n
    rng = np.random.default_rng()
    coreset_inds = rng.choice(np.arange(n), size=m, replace=False, p=weights_prob_dist)
    q_points = points[coreset_inds]
    q_weights = weights[coreset_inds] * float(n) / m
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

def semi_uniform_coreset(
    points,
    k,
    m,
    norm,
    kmeans_alg=cluster_pp_slow,
    allotted_time=np.inf,
    j_func='log',
    sample_method='sens',
    weights=None,
    **kwargs
):
    j_func_dict = {
        '2': 2,
        '10': 10,
        'log': np.log(k),
        'sqrt': np.sqrt(k),
        'half': k / 2
    }
    j = int(j_func_dict[j_func])
    if weights is None:
        weights = np.ones(len(points))
    centers, labels, costs = kmeans_alg(
        points,
        j,
        weights,
        allotted_time=allotted_time
    )

    sensitivities = bound_sensitivities(centers, labels, costs)
    r_points, r_weights, r_labels = get_coreset(sensitivities, m, points, labels, weights=weights)
        
    return r_points, r_weights, r_labels

def lightweight_coreset(
    points,
    k,
    m,
    norm,
    allotted_time=np.inf,
    weights=None,
    **kwargs
):
    if weights is None:
        weights = np.ones(len(points))
    center = np.zeros(len(points[0]))
    for point in points:
        for d in range(len(point)):
            center[d] += point[d]
    for d in range(len(center)):
        center[d] /= len(points)
    labels = np.ones(len(points))
    costs = np.zeros(len(points))
    for i, point in enumerate(points):
        for d in range(len(point)):
            costs[i] += (point[d] - center[d]) ** 2
        costs[i] = np.sqrt(costs[i])
        costs[i] = costs[i] ** norm
    costs = np.array(costs)
    costs *= weights

    sensitivities = bound_sensitivities([1], labels, costs, alpha=1)
    r_points, r_weights, r_labels = get_coreset(sensitivities, m, points, labels, weights=weights)

    return r_points, r_weights, r_labels

def stream_kmpp(
    points,
    weights,
    k,
    m,
    norm,
    hst_count_from_norm=True,
    **kwargs
):
    centers, labels, costs = fast_cluster_pp(
        points,
        int(m/2), # The kmeans++ algorithm samples 2k points to get an O(1) approximation. We want to sample m points exactly.
        norm=norm,
        weights=weights,
        hst_count_from_norm=hst_count_from_norm,
    )
    weights = np.zeros_like(centers)
    for i, center in tqdm(enumerate(centers), total=len(centers)):
        for label in labels.astype(np.int32):
            if label == center:
                weights[i] += 1

    return points[centers], weights, labels[centers]

def fast_coreset(
    points,
    weights,
    k,
    m,
    norm,
    hst_count_from_norm=True,
    allotted_time=np.inf,
    loud=False,
    **kwargs
):
    centers, labels, costs = fast_cluster_pp(
        points,
        k,
        norm=norm,
        weights=weights,
        hst_count_from_norm=hst_count_from_norm,
        allotted_time=allotted_time,
        loud=loud
    )
    if centers is None:
        # Ran out of time building the trees so we need to randomly initialize the data
        print('Ran out of time while building trees! Making coreset uniformly at random.')
        assert labels is None and costs is None
        return uniform_coreset(points, m)

    sensitivities = bound_sensitivities(centers, labels, costs)
    q_points, q_weights, q_labels = get_coreset(sensitivities, m, points, labels, weights=weights)
    return q_points, q_weights, q_labels

def evaluate_coreset(points, k, coreset, weights, point_weights=None):
    if point_weights is None:
        point_weights = np.ones(len(points))
    centers, _, _ = cluster_pp(coreset, k, weights=weights)

    coreset_assignments, coreset_costs = get_cluster_assignments(coreset, centers, coreset[centers])
    coreset_costs *= weights
    coreset_cost = np.sum(coreset_costs)

    dataset_assignments, dataset_costs = get_cluster_assignments(points, centers, coreset[centers])
    dataset_costs *= point_weights
    dataset_cost = np.sum(dataset_costs)
    if coreset_cost == 0:
        acc = np.inf
    else:
        acc = max(dataset_cost / coreset_cost, coreset_cost / dataset_cost)

    return acc

if __name__ == '__main__':
    g_norm = 2
    g_k = 100
    g_points, g_labels = get_dataset('nytimes', n_points=50000, D=50, num_centers=50, k=g_k, class_imbalance=5)

    g_m_scalar = 40
    g_allotted_time = 600
    g_hst_count_from_norm = True
    g_kmeans_alg = cluster_pp_slow
    g_points = jl_proj(g_points, g_k, eps=0.5)

    # method = 'fast'
    # method = 'lightweight'
    # method = 'semi_uniform'
    # method = 'sensitivity'
    # method = 'bico'
    method = 'uniform'

    start = time()
    g_weights = np.ones((len(g_points)))
    if method == 'fast':
        q_points, q_weights, q_labels = fast_coreset(
            g_points,
            g_weights,
            g_k,
            g_k * g_m_scalar,
            g_norm,
            hst_count_from_norm=g_hst_count_from_norm,
            allotted_time=g_allotted_time,
            loud=True
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
    elif method == 'lightweight':
        q_points, q_weights, q_labels = lightweight_coreset(
            g_points,
            g_k,
            g_k * g_m_scalar,
            g_norm,
            weights=g_weights,
        )
    elif method == 'semi_uniform':
        q_points, q_weights, q_labels = semi_uniform_coreset(
            g_points,
            g_k,
            g_k * g_m_scalar,
            g_norm,
            kmeans_alg=g_kmeans_alg,
            weights=g_weights,
        )
    elif method == 'bico':
        q_points, q_weights, q_labels = bico_coreset(g_points, g_k, g_k * g_m_scalar, g_allotted_time)
    else:
        q_points, q_weights, q_labels = uniform_coreset(g_points, g_k * g_m_scalar, weights=g_weights)
    end = time()
    print(end - start)
    print('Coreset cost ratio:', evaluate_coreset(g_points, g_k, q_points, q_weights))

    start = time()
    fast_points, fast_weights, fast_labels = fast_coreset(
        g_points,
        g_weights,
        g_k,
        g_k * g_m_scalar,
        g_norm,
        hst_count_from_norm=g_hst_count_from_norm,
        allotted_time=g_allotted_time,
        loud=True
    )
    print(time() - start)
    print('Coreset cost ratio:', evaluate_coreset(g_points, g_k, fast_points, fast_weights))

    # Visualize
    # model = PCA(n_components=2)
    # g_embedding = model.fit_transform(g_points)
    # q_embedding = model.transform(q_points)
    # fast_embedding = model.transform(fast_points)
    # fig, axes = plt.subplots(1, 3)

    # axes[0].scatter(g_embedding[:, 0], g_embedding[:, 1], c=g_labels, alpha=0.5, s=5)
    # axes[1].scatter(q_embedding[:, 0], q_embedding[:, 1], c=q_labels, alpha=0.5, s=5)
    # axes[2].scatter(fast_embedding[:, 0], fast_embedding[:, 1], c=fast_labels, alpha=0.5, s=5)

    # axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # axes[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # axes[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # axes[2].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # plt.show()
