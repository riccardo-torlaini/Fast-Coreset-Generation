import os
import copy
import numpy as np
import numba
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from kmeans_plusplus import cluster_pp
from kmeans_plusplus_slow import cluster_pp_slow
from experiment_utils.get_data import get_dataset
from experiment_utils.get_algorithm import get_algorithm, get_results
from make_coreset import semi_uniform_coreset, lightweight_coreset, evaluate_coreset, fast_coreset
from utils import jl_proj

def run_class_imbalance_experiments():
    norm = 2
    k = 100
    m_scalar = 40
    allotted_time = 600
    n_points = 50000
    weights = np.ones(n_points)
    num_centers = 50
    D = 50
    hst_count_from_norm = True
    kmeans_alg = cluster_pp

    runs = 5
    class_imbalances = [0, 1, 3, 5]
    cluster_sizes = [None for i in range(len(class_imbalances))]
    results = np.zeros((runs, len(class_imbalances), 5))
    for i, run in enumerate(range(runs)):
        print('\n\n')
        print('Run', run + 1)
        print('\n\n')
        for j, ci in enumerate(class_imbalances):
            print('\n')
            print('\tclass_imbalance:', ci)
            print('\n')
            points, labels = get_dataset('blobs', n_points=n_points, D=D, num_centers=num_centers, k=k, class_imbalance=ci)
            if cluster_sizes[j] is None:
                cluster_sizes[j] = list(np.unique(labels, return_counts=True)[1])
            else:
                cluster_sizes[j] += list(np.unique(labels, return_counts=True)[1])

            m = k * m_scalar

            q_points, q_weights, q_labels = lightweight_coreset(points, k, m, norm)
            distortion = evaluate_coreset(points, k, q_points, q_weights)
            print('\t\tCoreset cost ratio:', distortion)
            results[i, j, 0] = distortion
            print()

            for l, j_func in enumerate(['2', 'log', 'sqrt']):
                print('\t\tJ_FUNC:', j_func)
                q_points, q_weights, q_labels = semi_uniform_coreset(points, k, m, norm, kmeans_alg=kmeans_alg, j_func=j_func)
                distortion = evaluate_coreset(points, k, q_points, q_weights)
                results[i, j, l+1] = distortion
                print('\t\tCoreset cost ratio:', distortion)
                print()

            q_points, q_weights, q_labels = fast_coreset(points, weights, k, m, norm, kmeans_alg=kmeans_alg)
            distortion = evaluate_coreset(points, k, q_points, q_weights)
            print('Coreset cost ratio:', distortion)

            results[i, j, -1] = distortion

    plot_cluster_sizes(cluster_sizes, n_points, num_centers, class_imbalances)
    print(np.mean(results, axis=0))
    print(np.var(results, axis=0))

def plot_cluster_sizes(cluster_sizes, n_points, num_centers, class_imbalances):
    min_val = 0
    max_val = n_points / 5
    fig, axes = plt.subplots(4, 1)
    plt.rcParams.update({'text.usetex': True})
    for i, gamma in enumerate(class_imbalances):
        axes[i].hist(cluster_sizes[i], bins=20, range=[min_val, max_val], color='steelblue')
        axes[i].set_ylim([1, len(cluster_sizes[0])])
        axes[i].set_yscale('log')
        axes[i].set_ylabel(r'$\gamma =${}'.format(str(gamma)))
    for i in range(len(class_imbalances) - 1):
        axes[i].tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False
        )
    plt.show()

if __name__ == '__main__':
    run_class_imbalance_experiments()
