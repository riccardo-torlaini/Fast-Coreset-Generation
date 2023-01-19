import os
import copy
import numpy as np
import numba
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from kmeans_plusplus import cluster_pp
from experiment_utils.get_data import get_dataset
from experiment_utils.get_algorithm import get_algorithm, get_results
from make_coreset import sensitivity_coreset, fast_coreset, uniform_coreset

def get_experiment_params(default_values, norm, param, val):
    params = copy.copy(default_values)
    params[param] = val
    params['norm'] = norm
    params['m'] = params['k'] * params['m_scalar']
    return params

def run_sweeps():
    results = {}
    datasets = ['blobs', 'mnist']
    methods = ['fast_coreset', 'uniform_sampling', 'sens_sampling']

    # Only apply for Gaussian mixture model dataset
    n_points = 100000
    D = 50
    num_centers = 50

    # Default values for sweep parameters
    default_values = {
        'k': 100,
        'm_scalar': 50,
        'allotted_time': 600,
        'hst_count_from_norm': True,
    }

    sweep_params = {
        # Params to sweep for all coreset algorithms
        'k': [10, 40, 100],
        'm_scalar': [2, 5, 10, 20],
        'allotted_time': [10, 30, 60, 120, 360],
        # Params to sweep for fast_coreset algorithm
        'hst_count_from_norm': [True, False],
    }

    outputs_path = 'outputs'
    if not os.path.isdir(outputs_path):
        os.makedirs(outputs_path)

    pbar = tqdm(methods, total=len(methods))
    pbar_description = 'method --- {} ; dataset --- {} dataset ; norm --- {} norm ; param --- {}; value --- {}'
    for method in pbar:
        print('Method --- {}'.format(method))
        coreset_alg = get_algorithm(method)
        method_output_path = os.path.join(outputs_path, method)
        if not os.path.isdir(method_output_path):
            os.makedirs(method_output_path)
        for dataset in datasets:
            print('\tDataset --- {}'.format(dataset))
            points, _ = get_dataset(dataset, n_points, D, num_centers)
            n, d = points.shape # Will be different from n_points or D if dataset is real-world data

            # Get solution that we will evaluate coreset against
            uniform_weights = np.ones(n)
            one_approx_centers, _, one_approx_costs = cluster_pp(
                points,
                default_values['k'],
                uniform_weights,
            )

            dataset_output_path = os.path.join(method_output_path, dataset)
            if not os.path.isdir(dataset_output_path):
                os.makedirs(dataset_output_path)
            for norm in [1, 2]:
                print('\t\tNorm --- {}'.format(str(norm)))
                norm_output_path = os.path.join(dataset_output_path, str(norm))
                if not os.path.isdir(norm_output_path):
                    os.makedirs(norm_output_path)
                for param, vals in sweep_params.items():
                    print('\t\t\tParam --- {}'.format(param))
                    param_output_path = os.path.join(norm_output_path, param)
                    if not os.path.isdir(param_output_path):
                        os.makedirs(param_output_path)
                    for val in vals:
                        print('\t\t\t\tValue --- {}'.format(str(val)))
                        if param == 'k':
                            # If our parameter is k, we need to get a new kmeans++ solution to evaluate against
                            one_approx_centers, _, one_approx_costs = cluster_pp(points, val, uniform_weights)
                        val_output_path = os.path.join(param_output_path, str(val))
                        if not os.path.isdir(val_output_path):
                            os.makedirs(val_output_path)
                        # Update progress bar
                        pbar.set_description(pbar_description.format(method, dataset, str(norm), param, str(val)))

                        if param in ['hst_count_from_norm'] and method != 'fast_coreset':
                            print('Boolean parameters only apply to fast coreset algorithm. Continuing...\n')
                            continue

                        params = get_experiment_params(default_values, norm, param, val)
                        acc, time, q_points, q_weights = get_results(
                            points,
                            coreset_alg,
                            params,
                        )
                        print(acc, time)
                        
                        metric_results = {'acc': acc, 'time': time}
                        coreset_results = {'coreset_points': q_points, 'coreset_weights': q_weights}
                        np.save(os.path.join(val_output_path, 'metrics.npy'), metric_results)
                        np.save(os.path.join(val_output_path, 'coreset.npy'), coreset_results)

if __name__ == '__main__':
    run_sweeps()
