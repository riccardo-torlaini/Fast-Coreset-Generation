import numpy as np
from time import time
from tqdm import tqdm

from fast_kmeans_plusplus import fast_cluster_pp

def get_log_delta_testbed(n_points, original_points, copy_dist, num_copies):
    rounds = int((n_points - len(original_points)) / num_copies)
    single_sequence = np.zeros((rounds, 2))
    for r in range(rounds):
        single_sequence[r, 0] = copy_dist * np.power(0.5, r)

    copied_seqs = np.array([single_sequence + np.array([0, 1]) * copy_dist * i / num_copies for i in range(num_copies)])
    copied_seqs = np.concatenate(copied_seqs, axis=0)
    points = np.concatenate((copied_seqs, original_points), axis=0)
    return points

def run_log_delta_experiments():
    norm = 2
    k = 2
    m_scalar = 3
    m = k * m_scalar
    n_points = 50000

    runs = 3
    num_sequence_points = 30000
    rounds = [20, 30, 40, 50]
    results = np.zeros((runs, len(rounds)))
    np.random.seed(123)
    copy_dist = 1
    original_points = np.random.uniform(-copy_dist, copy_dist, (n_points, 2))
    fast_cluster_pp(original_points, k) # run fast-kmeans++ once since the first run is the slowest for some reason
    for j, r in enumerate(rounds):
        num_copies = int(num_sequence_points / r)
        points = get_log_delta_testbed(n_points=n_points, original_points=original_points[:(n_points - num_copies * r)], copy_dist=copy_dist, num_copies=num_copies)
        weights = np.ones(len(points))
        for i, run in enumerate(range(runs)):
            start = time()
            fast_cluster_pp(points, k)
            end = time()
            print(end - start)
            results[i, j] = end - start
    print(np.mean(results, axis=0))
    print(np.var(results, axis=0))

if __name__ == '__main__':
    run_log_delta_experiments()
