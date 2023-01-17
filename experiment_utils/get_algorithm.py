from time import time
import numpy as np
from make_coreset import fast_coreset, sensitivity_coreset, uniform_coreset, evaluate_coreset

ALG_DICT = {
    'sens_sampling': sensitivity_coreset,
    'fast_coreset': fast_coreset,
    'uniform_sampling': uniform_coreset
}

def get_algorithm(algorithm_type):
    return ALG_DICT[algorithm_type]

def get_results(
    points,
    coreset_alg,
    params,
    centers,
    costs
):
    start = time()
    q_points, q_weights, _ = coreset_alg(
        points,
        params['k'],
        params['eps'],
        params['norm'],
        double_k=params['double_k'],
        make_second_coreset=params['make_second_coreset'],
        hst_count_from_norm=params['hst_count_from_norm'],
        oversample=params['oversample']
    )
    end = time()
    acc = evaluate_coreset(points, params['k'], q_points, q_weights, centers, costs)
    return acc, end - start, q_points, q_weights
