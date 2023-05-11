from time import time
from tqdm.auto import tqdm
import numpy as np
from utils import get_cluster_assignments, get_all_dists_to_centers, get_min_dists_to_centers

def cluster_pp(points, k, weights, allotted_time=np.inf):
    n, d = int(points.shape[0]), int(points.shape[1])
    centers = [np.random.choice(n)]
    sq_dists = None
    start = time()
    elapsed_time = 0
    for i in tqdm(range(2*k), total=2*k):
        sq_dists = get_min_dists_to_centers(points, points[np.array(centers)[-1]], sq_dists)
        weighted_sq_dists = sq_dists * weights
        if np.sum(weighted_sq_dists) > 0:
            probs = weighted_sq_dists / np.sum(weighted_sq_dists)
        else:
            probs = np.ones_like(probs) / len(probs)
        centers.append(np.random.choice(n, p=probs))
        elapsed_time = time() - start
        if elapsed_time > allotted_time:
            break
    if elapsed_time > allotted_time:
        print('Ran out of time! Only processed {} of {} centers'.format(len(centers), k))
    centers = np.array(centers)
    assignments, costs = get_cluster_assignments(points, centers, points[centers])
    return centers, assignments, costs
