import numpy as np
import numba
from time import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from kmeans_plusplus import cluster_pp
from kmeans_plusplus_slow import cluster_pp_slow
from fast_kmeans_plusplus import fast_cluster_pp
from coreset_sandbox import HST, hst_dist, fit_tree, assert_hst_correctness
from multi_hst import make_multi_HST
from utils import bound_sensitivities, jl_proj
from make_coreset import make_rough_coreset, make_true_coreset, evaluate_coreset

def sweep_oversample(points, k, eps, norm, alpha):
    sweep_vals = [1, 2, 4, 8, 16, 32, 64]
    num_rounds = 5
    error_deltas = np.zeros((len(sweep_vals), num_rounds))
    times = np.zeros((len(sweep_vals), num_rounds))
    for i, oversample in enumerate(sweep_vals):
        for j in range(num_rounds):
            start = time()
            q_points, q_weights, _ = make_rough_coreset(points, k, eps, norm, alpha, oversample=oversample)
            q_points, q_weights, q_labels = make_true_coreset(q_points, q_weights, k, eps, norm, alpha)
            end = time()
            error_deltas[i, j] = evaluate_coreset(points, k, q_points, q_weights)
            times[i, j] = end - start

    error_means = np.mean(error_deltas, axis=1)
    error_var = np.std(error_deltas, axis=1)
    plt.xscale('log')
    plt.plot(sweep_vals, error_means)
    plt.fill_between(
        sweep_vals,
        error_means - error_var,
        error_means + error_var,
        alpha=0.5
    )
    plt.ylim(0, 3)
    plt.xlabel('Oversample scalar')
    plt.ylabel('Coreset cost ratio')
    plt.title('How much do we need to oversample by for rough coreset?')
    plt.show()
    plt.close()
    plt.clf()

    time_means = np.mean(times, axis=1)
    time_var = np.std(times, axis=1)
    plt.xscale('log')
    plt.plot(sweep_vals, time_means, c='r')
    plt.fill_between(
        sweep_vals,
        time_means - time_var,
        time_means + time_var,
        alpha=0.5,
        color='r'
    )
    plt.xlabel('Oversample scalar')
    plt.ylabel('Fast Coreset Construction time')
    plt.title('Does the oversample scalar affect coreset construction time?')
    plt.show()

def eval_double_k(points, eps, norm, alpha):
    k_vals = [10, 50, 100, 500, 1000]
    results = np.zeros((2, len(k_vals)))
    for i, k in enumerate(k_vals):
        for j, double_k in enumerate([0, 1]):
            q_points, q_weights, _ = make_rough_coreset(
                points,
                k,
                eps,
                norm,
                alpha,
                oversample=5,
                double_k=double_k
            )
            q_points, q_weights, q_labels = make_true_coreset(
                q_points,
                q_weights,
                k,
                eps,
                norm,
                alpha,
                double_k=double_k
            )
            results[j, i] = evaluate_coreset(g_points, k, q_points, q_weights)
    plt.xscale('log')
    plt.plot(k_vals, results[0, :], c='r')
    plt.plot(k_vals, results[1, :], c='b')
    plt.xlabel('k')
    plt.ylabel('coreset cost ratio')
    plt.title('Does using 2k for fast-kmeans++ matter? Red=k, blue=2k')
    plt.show()

def sweep_k(points, eps, norm, alpha):
    sweep_vals = [10, 50, 100, 500]
    oversample = 5
    weights = np.ones((len(points))) / len(points)
    error_deltas = np.zeros((len(sweep_vals)))
    our_times = np.zeros((len(sweep_vals)))
    orig_times = np.zeros((len(sweep_vals)))
    slow_times = np.zeros((len(sweep_vals)))
    for i, k in enumerate(sweep_vals):
        start = time()
        q_points, q_weights, _ = make_rough_coreset(points, k, eps, norm, alpha, oversample)
        # q_points, q_weights, q_labels = make_true_coreset(q_points, q_weights, k, eps, norm, alpha)
        end = time()
        error_deltas[i] = evaluate_coreset(points, k, q_points, q_weights)
        our_times[i] = end - start

        start = time()
        _, _, _ = make_true_coreset(points, weights, k, eps, norm, alpha)
        end = time()
        orig_times[i] = end - start

        start = time()
        _, _, _ = make_true_coreset(points, weights, k, eps, norm, alpha, kmeans_alg=cluster_pp_slow)
        end = time()
        slow_times[i] = end - start

    # plt.xscale('log')
    # plt.plot(sweep_vals, error_deltas)
    # plt.ylim(0, 3)
    # plt.xlabel('k')
    # plt.ylabel('Coreset cost ratio')
    # plt.title('Effect of k on coreset accuracy')
    # plt.show()
    # plt.close()
    # plt.clf()

    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plots = ax.plot(sweep_vals, our_times, c='r', label='fast-kmeans++')
    plots += ax.plot(sweep_vals, orig_times, c='m', label='numpy kmeans++')
    plots += ax.plot(sweep_vals, slow_times, c='k', label='python kmeans++')
    ax.set_xlabel('k')
    ax.set_ylabel('Fast Coreset Construction time')
    ax.set_title('Effect of k on runtime')
    labels = [plot.get_label() for plot in plots]

    legend = ax.legend(handles=plots, labels=labels)
    plt.show()


if __name__ == '__main__':
    n_points = 20000
    D = 1000
    num_centers = 10
    g_alpha = 10
    g_norm = 1
    g_points, _ = make_blobs(n_points, D, centers=num_centers)
    g_k = 40
    g_eps = 0.5
    g_points = jl_proj(g_points, g_k, g_eps)

    sweep_oversample(g_points, g_k, g_eps, g_norm, g_alpha)
    # eval_double_k(g_points, g_eps, g_norm, g_alpha)
    # sweep_k(g_points, g_eps, g_norm, g_alpha)

