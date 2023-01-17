import numpy as np
from sklearn.datasets import make_blobs

def get_dataset(dataset_type, n_points=1000, D=50, num_centers=10):
    if dataset_type == 'blobs':
        return make_blobs(n_points, D, centers=num_centers)
    elif dataset_type == 'single_blob':
        return make_blobs(n_points, D, centers=1)
    else:
        raise ValueError('Dataset not implemented')
