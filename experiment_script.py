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
from utils import jl_proj

def get_experiment_params(default_values, norm, param, val):
    params = copy.copy(default_values)
    params[param] = val
    params['norm'] = norm
    params['m'] = params['k'] * params['m_scalar']
    return params

def run_sweeps():
    results = {}
    datasets = [
        'artificial',
        'geometric',
        'benchmark',
        'blobs',
        'mnist',
        'adult',
        # 'song',
        # 'census',
        # 'cover_type'
    ]
    # methods = ['fast_coreset', 'semi_uniform', 'uniform_sampling', 'sens_sampling', 'lightweight', 'bico']
    # methods = ['fast_coreset', 'sens_sampling']
    methods = ['uniform_sampling', 'lightweight', 'semi_uniform']

    # Only apply for Gaussian mixture and 1-outlier datasets
    n_points = 50000
    D = 50
    num_centers = 50

    # Default values for sweep parameters on small datasets
    small_default_values = {
        'k': 100,
        'j_func': '2', # Only applies for semi-uniform coreset
        'sample_method': 'sens', # Only applies for semi-uniform coreset
        'm_scalar': 40,
        'composition': False,
        'allotted_time': 120,
        'hst_count_from_norm': True, # Only applies to fast-coreset
    }

    # Default values for sweep parameters on small datasets
    large_default_values = {
        'k': 500,
        'j_func': '2', # Only applies for semi-uniform coreset
        'sample_method': 'sens', # Only applies for semi-uniform coreset
        'composition': False,
        'm_scalar': 40,
        'allotted_time': 360,
        'hst_count_from_norm': True, # Only applies to fast-coreset
    }

    small_sweep_params = {
        # Params to sweep for all coreset algorithms
        # 'k': [50, 100, 200, 400],
        'composition': [False, True],
        # 'j_func': ['2', '10', 'log', 'sqrt'],
        # 'sample_method': ['sens', 'uniform'],
        # 'm_scalar': [20, 40, 60, 80],
        # 'allotted_time': [0, 0.5, 1, 3, 5, 7, 10, 20],
        # 'hst_count_from_norm': [True, False], # Only applies to fast_coreset algorithm
    }

    large_sweep_params = {
        # Params to sweep for all coreset algorithms
        'k': [100, 200, 400],
        'j_func': ['2', '10', 'log', 'sqrt'],
        # 'sample_method': ['sens', 'uniform'],
        'm_scalar': [20, 40, 60, 80],
        # 'allotted_time': [60, 120, 360],
        'hst_count_from_norm': [True, False], # Only applies to fast_coreset algorithm
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

            if n > 100000 and method == 'sens_sampling':
                # Don't run slow sensitivity sampling on large datasets
                continue

            # Get different default values and sweep parameters depending on dataset size
            if n > 100000:
                default_values = large_default_values
                sweep_params = large_sweep_params
            else:
                default_values = small_default_values
                sweep_params = small_sweep_params

            dataset_output_path = os.path.join(method_output_path, dataset)
            if not os.path.isdir(dataset_output_path):
                os.makedirs(dataset_output_path)

            # for norm in [1, 2]:
            for norm in [2]:
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
                        val_output_path = os.path.join(param_output_path, str(val))
                        if not os.path.isdir(val_output_path):
                            os.makedirs(val_output_path)
                        # Update progress bar
                        pbar.set_description(pbar_description.format(method, dataset, str(norm), param, str(val)))

                        if param in ['hst_count_from_norm'] and method != 'fast_coreset':
                            print('Boolean parameters only apply to fast coreset algorithm. Continuing...\n')
                            continue
                        if param in ['j_func', 'sample_method'] and method != 'semi_uniform':
                            print('{} only applies to semi-uniform algorithm. Continuing...\n'.format(param))
                            continue

                        params = get_experiment_params(default_values, norm, param, val)

                        if np.log(params['k']) / (0.5 ** 2) < points.shape[1]:
                            proj_points = jl_proj(points, params['k'], 0.5)
                        else:
                            proj_points = points

                        accuracies, times, q_points, q_weights = get_results(
                            proj_points,
                            coreset_alg,
                            params,
                        )

                        metric_results = {'acc': accuracies, 'time': times}
                        coreset_results = {'coreset_points': q_points, 'coreset_weights': q_weights}
                        np.save(os.path.join(val_output_path, 'metrics.npy'), metric_results)
                        np.save(os.path.join(val_output_path, 'coreset.npy'), coreset_results)

if __name__ == '__main__':
    run_sweeps()
