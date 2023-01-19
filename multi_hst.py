import numpy as np
from hst import hst_dist, fit_tree, assert_hst_correctness

### HST code ###
class MultiHST:
    def __init__(self, roots, ptc_dicts):
        self.roots = roots
        self.ptc_dicts = ptc_dicts

    def __len__(self):
        return len(self.roots)

def multi_tree_dist(multi_hst, a, b):
    min_dist = np.inf
    for i, (root, ptc_dict) in enumerate(zip(multi_hst.roots, multi_hst.ptc_dicts)):
        ptc_dict = multi_hst.ptc_dicts[i]
        dist = hst_dist(ptc_dict, a, b, root)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def assert_multi_hst_correctness(multi_hst, points):
    true_dist_squared = np.sqrt(np.sum(np.square(points[10] - points[30])))
    multi_dist_squared = multi_tree_dist(multi_hst, 10, 30)

    norm = len(multi_hst) - 1
    # If we are in the one-tree case, we don't want norm to be 0
    if norm == 0:
        norm = 1
    true_dist_squared = true_dist_squared ** norm
    multi_dist_squared = multi_dist_squared ** norm
    assert multi_dist_squared > true_dist_squared
    assert multi_dist_squared < 10 * true_dist_squared * int(points.shape[1]) ** norm

def make_multi_HST(points, k, eps, num_trees):
    roots, ptc_dicts = [], []
    for i in range(num_trees):
        root, ptc_dict = fit_tree(points)
        # assert_hst_correctness(root, ptc_dict, points)
        roots.append(root)
        ptc_dicts.append(ptc_dict)
    multi_hst = MultiHST(roots, ptc_dicts)
    # assert_multi_hst_correctness(multi_hst, points)
    return multi_hst
