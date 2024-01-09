import numpy as np
from kmeans_plusplus import cluster_pp
from experiment_utils.get_data import get_dataset
from utils import jl_proj
from make_coreset import \
    uniform_coreset, \
    sensitivity_coreset, \
    fast_coreset, \
    evaluate_coreset

def compare_distortions():
    datasets = [
        'mnist',
        'adult',
        'kitti',
        'star',
        'cover_type',
        'taxi',
    ]
    k = 100
    m = 40 * k
    iterations = 3
    for dataset in datasets:
        print('\n{}\n'.format(dataset))
        points, _ = get_dataset(dataset)
        # if np.log(k) / (0.25 ** 2) < points.shape[1]:
        #     points = jl_proj(points, k, eps=0.25)

        sens_distortions = []
        for i in range(iterations):
            sens_samples, sens_weights, _ = sensitivity_coreset(points, k, m, norm=2, kmeans_alg=cluster_pp)
            distortion = evaluate_coreset(points, k, sens_samples, sens_weights)
            sens_distortions.append(distortion)
        sens_distortion = np.mean(sens_distortions)
        print('Sens. distortion on {} dataset: {}'.format(dataset, sens_distortion))

        unif_distortions = []
        for i in range(iterations):
            unif_samples, unif_weights, _ = uniform_coreset(points, m)
            distortion = evaluate_coreset(points, k, unif_samples, unif_weights)
            unif_distortions.append(distortion)
        unif_distortion = np.mean(unif_distortions)
        unif_distortion_ratio = unif_distortion / sens_distortion
        print('Uniform Distortion Ratio:', unif_distortion_ratio)

        weights = np.ones(len(points))
        fast_distortions = []
        for i in range(iterations):
            fast_samples, fast_weights, _ = fast_coreset(points, weights, k, m, norm=2)
            distortion = evaluate_coreset(points, k, fast_samples, fast_weights)
            fast_distortions.append(distortion)
        fast_distortion = np.mean(fast_distortions)
        fast_distortion_ratio = fast_distortion / sens_distortion
        print('Fast-Coreset Distortion Ratio:', fast_distortion_ratio)



if __name__ == '__main__':
    compare_distortions()
