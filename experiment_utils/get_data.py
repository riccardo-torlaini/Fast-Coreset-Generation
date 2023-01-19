import numpy as np
from sklearn.datasets import make_blobs
import os
import subprocess
from mnist import MNIST

def get_dataset(dataset_type, n_points=1000, D=50, num_centers=10, k=20):
    if dataset_type == 'blobs':
        return make_blobs(n_points, D, centers=num_centers)
    elif dataset_type == 'single_blob':
        return make_blobs(n_points, D, centers=1)
    elif dataset_type == 'artificial':
        return get_artificial_dataset(n_points, D)
    elif dataset_type == 'mnist':
        return load_mnist()
    elif dataset_type == 'benchmark':
        return load_coreset_benchmark(k)
    else:
        raise ValueError('Dataset not implemented')

def get_artificial_dataset(n_points, D):
    g_points = np.ones([n_points, D])
    g_points[0] = -1000 * np.ones(D)
    g_points += np.random.rand(n_points, D)
    return g_points, np.ones(n_points)

def load_mnist():
    mnist_data_path = os.path.join('data', 'mnist')
    if not os.path.isdir(mnist_data_path):
        subprocess.call(os.path.join('experiment_utils', 'get_mnist.sh'))

    mndata = MNIST(mnist_data_path)
    points, labels = mndata.load_training()
    points = np.array(points, dtype=np.float32)
    labels = np.array(labels)
    return points, labels

def load_coreset_benchmark(k, alpha=3):
    # Arbitrarily split up k into three sections with denominators that are mutually prime
    k_one = int(k / 5)
    k_two = int((k - k_one) / 3)
    k_three = int(k - k_one - k_two)
    benchmark_matrices = []
    for k in [k_one, k_two, k_three]:
        matrix = -1 * np.ones([k, k]) / k
        matrix += np.eye(k)

        columns = [matrix]
        for a in range(1, alpha):
            matrix_sections = np.split(columns[-1], k, axis=0)
            stretched_matrix = []
            for section in matrix_sections:
                for copy in range(k):
                    stretched_matrix.append(section)
            stretched_matrix = np.squeeze(np.concatenate(stretched_matrix, axis=0))
            columns.append(stretched_matrix)
        num_rows = len(columns[-1])
        
        equal_size_columns = []
        for column in columns:
            num_copies = int(num_rows / len(column))
            equal_size_column = np.concatenate([column for _ in range(num_copies)], axis=0)
            equal_size_columns.append(equal_size_column)
        benchmark_matrix = np.concatenate(equal_size_columns, axis=1)

        # Add a random vector offset to each benchmark dataset
        offset = np.sin(k) * k ** 2
        benchmark_matrix += offset

        benchmark_matrices.append(benchmark_matrix)

    padded_benchmarks = []
    sizes = [bm.shape[1] for bm in benchmark_matrices]
    for i, benchmark_matrix in enumerate(benchmark_matrices):
        pad_amount = max(sizes) - sizes[i]
        npad = ((0, 0), (0, pad_amount))
        padded = np.pad(benchmark_matrix, pad_width=npad, mode='constant', constant_values=0)
        padded_benchmarks.append(padded)

    padded_benchmarks = np.concatenate(padded_benchmarks, axis=0)
    return padded_benchmarks, np.ones(len(padded_benchmarks))
