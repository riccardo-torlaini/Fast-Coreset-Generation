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
    bico_coreset

ALG_DICT = {
    'sens_sampling': sensitivity_coreset,
    'fast_coreset': fast_coreset,
    'uniform_sampling': uniform_coreset,
    'semi_uniform': semi_uniform_coreset,
    'lightweight': lightweight_coreset,
    'bico': bico_coreset
}

def get_algorithm(algorithm_type):
    return ALG_DICT[algorithm_type]

def run_composition_experiments(
    points,
    coreset_alg,
    params,
    num_tree_layers=4,
    class_imbalance=3
):
    partition_coresets = []
    partition_weights = []
    n = len(points)
    num_eq_classes = np.power(2, num_tree_layers)
    partition_size = int(n / num_eq_classes)

    points = points[np.random.permutation(n)]
    partition = [points[partition_size*i:partition_size*(i+1)] for i in range(num_eq_classes)]
    print([partition[i].shape for i in range(num_eq_classes)])
    quit()
    start = 0
    for cluster_size in partition:
        subset = proj_points[start:start+cluster_size]
        if not len(subset):
            continue
        start += cluster_size
        _, _, q_points, q_weights = get_results(subset, coreset_alg, params, iterations=1)
        partition_coresets.append(q_points)
        partition_weights.append(q_weights)

def get_results(
    points,
    coreset_alg,
    params,
    iterations=10
):
    times = np.zeros(iterations)
    accuracies = np.zeros(iterations)
    print('Iteration:', end=' ')
    for i in range(iterations):
        print(i, end=' ')
        points_copy = np.copy(points)
        if params['composition']:
            times[i], q_points, q_weights = run_composition_experiments(points, coreset_alg, params)
        else:
            start = time()
            q_points, q_weights, _ = coreset_alg(
                points,
                k=params['k'],
                j_func=params['j_func'],
                m=params['m'],
                norm=params['norm'],
                hst_count_from_norm=params['hst_count_from_norm'],
                allotted_time=params['allotted_time'],
            )
            end = time()
        assert np.allclose(points, points_copy)
        times[i] = end - start
        acc = evaluate_coreset(
            points,
            k=params['k'],
            coreset=q_points,
            weights=q_weights,
        )
        accuracies[i] = acc
    return accuracies, times, q_points, q_weights
