from tqdm import tqdm
import csv
import os
import subprocess

import cv2 as cv
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.random_projection import SparseRandomProjection
from sklearn.datasets import make_blobs
from mnist import MNIST

def normalize(points):
    points -= np.min(points, axis=0, keepdims=True)
    points /= np.max(points, axis=0, keepdims=True)
    return points


def get_dataset(dataset_type, n_points=1000, D=50, num_centers=10, k=50, class_imbalance=5.0):
    if dataset_type == 'blobs':
        return get_blobs_dataset(n_points, D, num_centers, class_imbalance=class_imbalance)
    elif dataset_type == 'artificial':
        return get_artificial_dataset(n_points, D)
    elif dataset_type == 'benchmark':
        return load_coreset_benchmark(k)
    elif dataset_type == 'geometric':
        return get_geometric_dataset(k, num_rounds=k)
    elif dataset_type == 'mnist':
        return load_mnist()
    elif dataset_type == 'adult':
        return load_adult_data()
    elif dataset_type == 'song':
        return load_song()
    elif dataset_type == 'cover_type':
        return load_cover_type()
    elif dataset_type == 'kdd_cup':
        return load_darpa_kdd_cup()
    elif dataset_type == 'census':
        return load_census_data()
    elif dataset_type == 'nytimes':
        return load_nytimes_data()
    elif dataset_type == 'caltech':
        return load_caltech_data()
    elif dataset_type == 'fraud':
        return load_fraud_data()
    elif dataset_type == 'taxi':
        return load_taxi_data()
    elif dataset_type == 'star':
        return load_star_data()
    elif dataset_type == 'kitti':
        return load_kitti_data()
    else:
        raise ValueError('Dataset not implemented')

def get_imbalanced_partition(n_points, num_eq_classes, class_imbalance=5.0):
    points_remaining = n_points
    cluster_sizes = np.zeros(num_eq_classes, dtype=np.int32)
    for i in range(num_eq_classes):
        mean_num_points = points_remaining / (num_eq_classes - i)
        size_scalar = np.exp((np.random.rand() - 0.5) * class_imbalance)
        cluster_size = mean_num_points * size_scalar
        cluster_sizes[i] = int(min(cluster_size, points_remaining))
        points_remaining = n_points - np.sum(cluster_sizes)
    return cluster_sizes

def get_blobs_dataset(n_points, D, num_centers, scalar=1000, class_imbalance=5.0, var=500):
    # 1) Get random sizes for each cluster
    cluster_sizes = get_imbalanced_partition(n_points, num_centers, class_imbalance)

    # 2) Put each cluster at a random position on the unit hypersphere
    points = []
    labels = []
    for i in range(num_centers):
        cluster_vector = np.random.multivariate_normal(
            mean=np.zeros(D),
            cov=np.eye(D),
            size=(1)
        )
        # cluster_vector /= np.linalg.norm(cluster_vector[0])
        points.append(np.ones((int(cluster_sizes[i]), D)) * cluster_vector)
        labels.append(np.ones(int(cluster_sizes[i])) * i)
    points = np.concatenate(points, axis=0)
    labels = np.concatenate(labels, axis=0)

    # 3) Zero-mean the dataset
    points -= np.mean(points, axis=0, keepdims=True)

    # 4) Scale the dataset
    points *= scalar
    points += np.random.multivariate_normal(
        mean=np.zeros(D),
        cov=np.eye(D) * var,
        size=(len(points))
    )
    return points, labels


def get_geometric_dataset(k, num_rounds, size_scalar=100):
    points = []
    for i in range(num_rounds):
        cluster = np.zeros((int(size_scalar * k / (2 ** i)), num_rounds))
        cluster[:, i] = 1
        points.append(cluster)
    points = np.concatenate(points, axis=0)
    points += np.random.rand(len(points), num_rounds) / 1000
    return points, np.ones(points.shape[0])

def get_artificial_dataset(n_points, D, num_outliers=5):
    g_points = np.ones([n_points, D])
    for outlier in range(num_outliers):
        g_points[outlier] = -1000 * np.random.rand(D)
    g_points += np.random.rand(n_points, D)
    return g_points, np.ones(n_points)

def remap_features(points, columns):
    """ Move from categorical inputs to continuous, normalized data """
    new_points = np.zeros(points.shape)
    for c in range(columns):
        column = points[:, c]
        _, column_remap = np.unique(column, return_inverse=True)
        column_remap = column_remap.astype(np.float32)
        # if np.max(column_remap) != 0:
        #     column_remap /= np.max(column_remap)
        new_points[:, c] = column_remap
    return new_points

def load_categorical_dataset(
    data_path,
    pickled_path,
    rows,
    columns,
    data_type,
    start_index=0,
    skip_first_row=False,
    end_index=None
):
    if os.path.exists(pickled_path):
        dataset = np.load(pickled_path, allow_pickle=True)[()]
        return dataset['points'], dataset['labels']
    print('Could not find pickled dataset at {}. Loading from txt file and pickling...'.format(pickled_path))

    points = np.empty([rows, columns], dtype=data_type)
    labels = np.zeros([rows])
    if end_index is None:
        end_index = columns
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        for i, line in tqdm(enumerate(reader), total=rows):
            points[i] = np.array(line)[start_index:end_index]
    if skip_first_row:
        points = points[1:]
        rows -= 1

    points = remap_features(points, columns)
    # If there are duplicate points, the HST will recur until segmentation fault
    points += np.random.rand(rows, columns) / 100000
    _, labels = np.unique(labels, return_inverse=True)
    np.save(pickled_path, {'points': points, 'labels': labels})
    return points, labels

def load_adult_data():
    # Adult census data found at https://archive.ics.uci.edu/ml/datasets/census+income
    directory = os.path.join('data', 'adult')
    data_path = os.path.join(directory, 'adult.data')
    pickled_path = os.path.join(directory, 'pickled_adult.npy')
    rows, columns = 48842, 14
    data_type = str
    end_index = -1
    return load_categorical_dataset(
        data_path,
        pickled_path,
        rows,
        columns,
        data_type,
        end_index=end_index
    )

def load_census_data():
    # https://archive.ics.uci.edu/ml/datasets/US+Census+Data+(1990)
    directory = os.path.join('data', 'census')
    data_path = os.path.join(directory, 'USCensus1990.data.txt')
    pickled_path = os.path.join(directory, 'pickled_census.npy')
    rows, columns = 2458286, 68
    data_type = str
    return load_categorical_dataset(
        data_path,
        pickled_path,
        rows,
        columns,
        data_type,
        skip_first_row=True
    )

def load_darpa_kdd_cup():
    # 1999 KDD Cup dataset found at
    # https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data?resource=download 
    directory = os.path.join('data', 'kdd_cup')
    data_path = os.path.join(directory, 'kddcup.data_10_percent', 'kddcup.data_10_percent')
    pickled_path = os.path.join(directory, 'pickled_kdd_cup.npy')
    rows, columns = 494021, 41
    data_type = str
    end_index = -1
    return load_categorical_dataset(
        data_path,
        pickled_path,
        rows,
        columns,
        data_type,
        end_index=end_index
    )

def load_cover_type():
    # Cover type dataset found at https://archive.ics.uci.edu/ml/datasets/covertype
    directory = os.path.join('data', 'cover_type')
    data_path = os.path.join(directory, 'covtype.data')
    pickled_path = os.path.join(directory, 'pickled_cover_type.npy')
    rows, columns = 581012, 54
    data_type = np.float32
    start_index = 1
    return load_categorical_dataset(
        data_path,
        pickled_path,
        rows,
        columns,
        data_type,
        start_index=start_index
    )

def load_caltech_data():
    # 101 images dataset found in https://data.caltech.edu/records/mzrjq-6wc02
    # We find the SIFT features of every image and put the total collection of sift features into a data matrix
    #   - this results in a 4.1 million points in 128 dimensions
    directory = os.path.join('data', 'caltech', 'caltech-101')
    pickled_path = os.path.join(directory, 'pickled_caltech.npy')
    if os.path.exists(pickled_path):
        dataset = np.load(pickled_path, allow_pickle=True)[()]
        return dataset['points'], dataset['labels']

    directory = os.path.join(directory, '101_ObjectCategories')
    points = []
    for i, class_path in tqdm(enumerate([d[0] for d in os.walk(directory)][1:]), total=101):
        label = os.path.split(class_path)[-1]
        walk = os.walk(class_path)
        for j, image_name in enumerate(list(walk)[0][2]):
            image_path = os.path.join(class_path, image_name)
            img = cv.imread(image_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            sift = cv.SIFT_create()
            _, descriptors = sift.detectAndCompute(gray, None)
            if descriptors is not None:
                points.append(descriptors)

    points = np.concatenate(points, 0)
    points = np.unique(points, axis=0)
    labels = np.arange(len(points))
    np.save(pickled_path, {'points': points, 'labels': labels})
    return points, labels

def load_nytimes_data():
    # NYTimes bag-of-words dataset from https://archive.ics.uci.edu/dataset/164/bag+of+words
    directory = os.path.join('data', 'nytimes')
    pickled_path = os.path.join(directory, 'pickled_nytimes.npy')
    if os.path.exists(pickled_path):
        dataset = np.load(pickled_path, allow_pickle=True)[()]
        return dataset['points'], dataset['labels']

    n_words = 102660
    n_docs = 300000
    proj_dim = 128
    high_dim_points = np.zeros([n_docs, n_words])
    points = np.zeros([n_docs, 128])
    with open(os.path.join(directory, 'docword.nytimes.txt')) as f:
        for i, line in tqdm(enumerate(f), total=69679427):
            if i < 3:
                continue
            line_vals = [int(i) for i in line.rstrip().split(' ')]
            high_dim_points[line_vals[0] - 1, line_vals[1] - 1] = line_vals[2]
    projector = SparseRandomProjection(128)
    step = 1000
    for i in tqdm(range(0, n_docs, step), total=n_docs/step):
        points[i:i+step] = projector.fit_transform(high_dim_points[i:i+step])
    points = np.unique(points, axis=0)
    print(points.shape)
    labels = np.arange(len(points))
    np.save(pickled_path, {'points': points, 'labels': labels})
    return points, labels

def load_fraud_data():
    # Downloaded from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/
    directory = os.path.join('data', 'fraud')
    pickled_path = os.path.join(directory, 'pickled_fraud.npy')
    if os.path.exists(pickled_path):
        dataset = np.load(pickled_path, allow_pickle=True)[()]
        return dataset['points'], dataset['labels']

    data_path = os.path.join('data', 'fraud', 'creditcard.csv')
    rows, columns = 284807, 28
    points = np.zeros([rows, columns])
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        for i, line in tqdm(enumerate(reader), total=rows):
            if i == 0:
                # First line is junk
                continue
            points[i-1] = [float(p) for p in line[1:29]]

    points = np.unique(points, axis=0)
    points = normalize(points)

    labels = np.arange(len(points))
    np.save(pickled_path, {'points': points, 'labels': labels})
    return points, labels

def load_song():
    # Million song dataset found at https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd
    directory = os.path.join('data', 'song')
    pickled_path = os.path.join(directory, 'pickled_song.npy')
    if os.path.exists(pickled_path):
        dataset = np.load(pickled_path, allow_pickle=True)[()]
        return dataset['points'], dataset['labels']
    print('Could not find pickled dataset at {}. Loading from txt file and pickling...'.format(pickled_path))

    data_path = os.path.join('data', 'song', 'YearPredictionMSD.txt')
    rows, columns = 515345, 90
    points = np.zeros([rows, columns])
    labels = np.zeros([rows])
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        for i, line in tqdm(enumerate(reader), total=rows):
            points[i] = [float(p) for p in line[1:]]
            labels[i] = line[0]

    _, labels = np.unique(labels, return_inverse=True)
    points = np.unique(points, axis=0)
    points = normalize(points)

    np.save(pickled_path, {'points': points, 'labels': labels})
    return points, labels

def load_star_data():
    directory = os.path.join('data', 'star')
    pickled_path = os.path.join(directory, 'pickled_star.npy')
    if os.path.exists(pickled_path):
        dataset = np.load(pickled_path, allow_pickle=True)[()]
        return dataset['points'], dataset['labels']
    print('Could not find pickled dataset at {}. Loading from csv file and pickling...'.format(pickled_path))

    from PIL import Image
    star_image = Image.open(os.path.join(directory, 'star.jpg'))
    base_width = 500
    wpercent = base_width / float(star_image.size[0])
    hsize = int(star_image.size[1] * wpercent)
    star_image = star_image.resize((base_width, hsize), Image.Resampling.LANCZOS)
    points = np.reshape(star_image, [-1, 3]).astype(np.float32)
    points = normalize(points)
    print(points.shape)

    # Can't have duplicate points for hst embeddings
    points += (np.random.rand(*points.shape) - 0.5) * 0.0001 

    labels = np.ones(len(points))
    np.save(pickled_path, {'points': points, 'labels': labels})

    return points, labels

def load_taxi_data():
    directory = os.path.join('data', 'taxi')
    pickled_path = os.path.join(directory, 'pickled_taxi.npy')
    if os.path.exists(pickled_path):
        dataset = np.load(pickled_path, allow_pickle=True)[()]
        return dataset['points'], dataset['labels']
    print('Could not find pickled dataset at {}. Loading from csv file and pickling...'.format(pickled_path))


    taxi_dataset = pd.read_csv(os.path.join(directory, 'train.csv'))
    str_locations = taxi_dataset['POLYLINE']
    points = []
    for i, str_location in enumerate(tqdm(str_locations, total=len(str_locations))):
        first_comma = str_location.find(',')
        first_bracket = str_location.find(']')
        try:
            x = float(str_location[2:first_comma])
            y = float(str_location[first_comma+1:first_bracket])
            points.append([x, y])
        except ValueError:
            continue

    points = np.array(points)
    points = np.unique(points, axis=0)
    points = normalize(points)
    labels = np.ones(len(points))
    np.save(pickled_path, {'points': points, 'labels': labels})

    return points, labels

def load_kitti_data():
    directory = os.path.join('data', 'kitti')
    pickled_path = os.path.join(directory, 'pickled_kitti.npy')
    if os.path.exists(pickled_path):
        dataset = np.load(pickled_path, allow_pickle=True)[()]
        return dataset['points'], dataset['labels']
    print('Could not find pickled dataset at {}. Loading from csv file and pickling...'.format(pickled_path))


    dataset_path = os.path.join(directory, 'data.h5')
    import pykitti
    dataset = pykitti.raw(directory, '2011_09_26', '0001')
    velo = np.array(list(dataset.velo))
    points = velo[0][:, :3]

    points = normalize(points)
    labels = np.ones(len(points))

    np.save(pickled_path, {'points': points, 'labels': labels})
    return points, labels


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
    padded_benchmarks += np.random.rand(padded_benchmarks.shape[0], padded_benchmarks.shape[1]) / 1000
    return padded_benchmarks.astype(np.float32), np.ones(len(padded_benchmarks))
