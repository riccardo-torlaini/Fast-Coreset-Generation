import numpy as np
from sklearn.datasets import make_blobs
import os
import subprocess
from mnist import MNIST

def get_dataset(dataset_type, n_points=1000, D=50, num_centers=10):
    if dataset_type == 'blobs':
        return make_blobs(n_points, D, centers=num_centers)
    elif dataset_type == 'single_blob':
        return make_blobs(n_points, D, centers=1)
    elif dataset_type == 'artificial':
        return get_artificial_dataset(n_points, D)
    elif dataset_type == 'mnist':
        return load_mnist()
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

