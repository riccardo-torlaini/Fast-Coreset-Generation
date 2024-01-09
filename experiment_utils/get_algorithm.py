from time import time
import numpy as np
from experiment_utils.get_data import get_dataset, get_imbalanced_partition
from make_coreset import \
    fast_coreset, \
    sensitivity_coreset, \
    uniform_coreset, \
    evaluate_coreset, \
    semi_uniform_coreset, \
    lightweight_coreset, \
    bico_coreset, \
    stream_kmpp

RANDOM_DATASETS = ['artificial', 'geometric', 'benchmark', 'blobs']

ALG_DICT = {
    'sens_sampling': sensitivity_coreset,
    'fast_coreset': fast_coreset,
    'uniform_sampling': uniform_coreset,
    'semi_uniform': semi_uniform_coreset,
    'lightweight': lightweight_coreset,
    'bico': bico_coreset,
    'stream_kmpp': stream_kmpp
}

def get_algorithm(algorithm_type):
    return ALG_DICT[algorithm_type]

def call_coreset_alg(coreset_alg, points, params, weights=None):
    if weights is None:
        weights = np.ones(len(points))
    return coreset_alg(
        points,
        weights=weights,
        k=params['k'],
        j_func=params['j_func'],
        m=params['m'],
        norm=params['norm'],
        hst_count_from_norm=params['hst_count_from_norm'],
        allotted_time=params['allotted_time'],
    )

def recursively_compose_coresets(leaf_coresets, leaf_weights, coreset_alg, params):
    if leaf_coresets.shape[0] == 1:
        q_points, q_weights, _ = call_coreset_alg(coreset_alg, leaf_coresets[0], params, leaf_weights[0])
        return q_points, q_weights

    num_leaves = int(len(leaf_coresets) / 2)
    assert num_leaves == len(leaf_weights) / 2
    left_coreset, left_weights = recursively_compose_coresets(
        leaf_coresets[:num_leaves],
        leaf_weights[:num_leaves],
        coreset_alg,
        params
    )
    right_coreset, right_weights = recursively_compose_coresets(
        leaf_coresets[num_leaves:],
        leaf_weights[num_leaves:],
        coreset_alg,
        params
    )
    points = np.concatenate([left_coreset, right_coreset], axis=0)
    weights = np.concatenate([left_weights, right_weights], axis=0)
    q_points, q_weights, _ = call_coreset_alg(coreset_alg, points, params, weights)
    return q_points, q_weights


def run_composition_experiments(points, coreset_alg, params, num_tree_layers=4):
    partition_coresets = []
    partition_weights = []
    n = len(points)
    num_eq_classes = np.power(2, num_tree_layers)
    partition_size = int(n / num_eq_classes)

    points = points[np.random.permutation(n)]
    partition = [points[partition_size*i:partition_size*(i+1)] for i in range(num_eq_classes)]
    for points_subset in partition:
        q_points, q_weights, _ = call_coreset_alg(coreset_alg, points_subset, params)
        partition_coresets.append(q_points)
        partition_weights.append(q_weights)
    partition_coresets = np.array(partition_coresets)
    partition_weights = np.array(partition_weights)

    composition_scales = np.array([1] + [np.power(2, i) for i in range(num_tree_layers)])
    handled_coresets = 0
    composed_coresets = []
    composed_weights = []
    for subtree_size in composition_scales:
        leaf_coresets = np.array(partition_coresets[handled_coresets:handled_coresets+subtree_size])
        leaf_weights = np.array(partition_weights[handled_coresets:handled_coresets+subtree_size])
        if len(leaf_coresets) == params['m']:
            subtree_coreset, subtree_weights = leaf_coresets[0], leaf_weights[0]
        else:
            subtree_coreset, subtree_weights = recursively_compose_coresets(
                leaf_coresets,
                leaf_weights,
                coreset_alg,
                params
            )
        composed_coresets.append(subtree_coreset)
        composed_weights.append(subtree_weights)
        handled_coresets += subtree_size

    composed_coresets = np.concatenate(composed_coresets, axis=0)
    composed_weights = np.concatenate(composed_weights, axis=0)
    q_points, q_weights, _ = call_coreset_alg(coreset_alg, composed_coresets, params, composed_weights)
    return q_points, q_weights


def get_results(
    coreset_alg,
    params,
    dataset,
    n_points=50000,
    D=50,
    num_centers=50,
):
    iterations = params['iterations']
    times = np.zeros(iterations)
    accuracies = np.zeros(iterations)
    for i in range(iterations):
        if dataset in RANDOM_DATASETS or i == 0:
            # Load in dataset each time if it's random and once if it's not
            points, _ = get_dataset(dataset, n_points, D, num_centers)
            np.random.shuffle(points)
        start = time()
        if params['composition']:
            q_points, q_weights = run_composition_experiments(points, coreset_alg, params)
        else:
            start = time()
            q_points, q_weights, _ = call_coreset_alg(coreset_alg, points, params)
        times[i] = time() - start
        acc = evaluate_coreset(
            points,
            k=params['k'],
            coreset=q_points,
            weights=q_weights,
        )
        print(acc)
        accuracies[i] = acc
    return accuracies, times, q_points, q_weights
