import numpy as np
import matplotlib as plt
from sklearn.random_projection import SparseRandomProjection
import sys
sys.setrecursionlimit(5000)

class HST:
    def __init__(self, P, delta=None, split_d=0, prev_split=None, root=True, depth=0, cell_id=0):
        """
        Initialize Hierarchically Separated Tree
        input:
            P : numpy array, shape (n x d)
        """
        self.P = P
        self.n, self.d = P.shape
        self.depth = depth
        self.max_depth = 0
        self.cell_id = cell_id
        self.root = root # Only true if this is the root of the tree

        self.parent = None
        self.left_child = None
        self.right_child = None

        if self.root:
            self.get_delta()
            self.random_shift()

    def get_delta(self):
        """ Get maximum distance in P along one axis """
        axis_dists = np.array([np.max(self.P[:, i]) - np.min(self.P[:, i]) for i in range(1, self.d)])
        self.delta = np.max(axis_dists)

    def random_shift(self):
        """ Apply a random shift to P so that all points are in the [0, 2 * Delta]^d box """
        # Move pointset to all-positive values
        self.P[:, 1:] -= np.min(self.P[:, 1:], axis=1, keepdims=True)
        self.P[:, 1:] += 1e-3

        # Apply a random shift in [0, delta]
        self.P[:, 1:] += np.random.random([1, self.d - 1]) * self.delta

def get_split_vars(split_d, delta, d):
    next_d = split_d + 1
    next_delta = delta
    if next_d >= d + 1:
        next_d = 1
        next_delta = delta / 2

    return next_d, next_delta

def fit_tree(P):
    point_to_cell_dict = {}
    root = HST(P)
    g_split_d = 1
    g_delta = root.delta
    g_prev_split = np.zeros([root.d + 1])

    def _fit(node, split_d, delta, prev_split):
        if len(node.P) == 1:
            node.max_depth = node.depth
            point_to_cell_dict[node.P[0, 0]] = node
            return node.cell_id + 1

        left_inds = node.P[:, split_d] <= (prev_split[split_d] + delta)
        left_P = node.P[left_inds]
        right_P = node.P[np.logical_not(left_inds)]

        next_d, next_delta = get_split_vars(split_d, delta, node.d)

        if len(left_P) >= 1:
            left_split = np.copy(prev_split)
            node.left_child = HST(left_P, root=False, depth=node.depth+1, cell_id=node.cell_id)
            node.cell_id = _fit(node.left_child, next_d, next_delta, left_split)
            node.left_child.parent = node
            left_depth = node.left_child.max_depth
        else:
            left_depth = 0

        if len(right_P) >= 1:
            right_split = np.copy(prev_split)
            right_split[split_d] += delta
            node.right_child = HST(right_P, root=False, depth=node.depth+1, cell_id=node.cell_id)
            node.cell_id = _fit(node.right_child, next_d, next_delta, right_split)
            node.right_child.parent = node
            right_depth = node.right_child.max_depth
        else:
            right_depth = 0

        node.max_depth = max(left_depth, right_depth)
        return node.cell_id

    _fit(root, g_split_d, g_delta, g_prev_split)
    return root, point_to_cell_dict

def tree_dist(ptc_dict, a, b, root):
    """
    root -- base of the tree
    a, b -- two points that we want the tree-distance between
    """
    max_depth = root.max_depth
    delta = root.delta
    cell_a = ptc_dict[a]
    depth_a = cell_a.depth
    cell_b = ptc_dict[b]
    depth_b = cell_b.depth

    even_odd = 0
    while cell_a.cell_id != cell_b.cell_id:
        if cell_a.depth == 0 and cell_b.depth == 0:
            raise ValueError('Went all the way up to root without paths intersecting')
        if cell_a.depth > 0 and even_odd % 2 == 0:
            cell_a = cell_a.parent
        if cell_b.depth > 0 and even_odd % 2 == 1:
            cell_b = cell_b.parent
        even_odd = (even_odd + 1) % 2

    print(cell_a.depth, cell_b.depth)
    print(depth_a, depth_b)


if __name__ == '__main__':
    P = np.random.randn(500, 1000)
    k = 10
    eps = 1e-1
    d = np.ceil(np.log(k) / (eps ** 2)).astype(np.int32)
    jl_proj = SparseRandomProjection(d)

    P = jl_proj.fit_transform(P)
    indices = np.expand_dims(np.arange(len(P)), -1)
    P = np.concatenate((indices, P), axis=1)
    root, ptc_dict = fit_tree(P)

    tree_dist(ptc_dict, 10, 30, root)
