import numpy as np
import matplotlib as plt
from sklearn.random_projection import SparseRandomProjection
import sys
sys.setrecursionlimit(5000)

class ListHST:
    def __init__(self, P):
        """
        Initialize Hierarchically Separated Tree
        input:
            P : numpy array, shape (n x d)
        """
        self.P = P
        self.n, self.d = P.shape
        self.layers = [[self.P]]
        self.corners = [[np.zeros([self.n])]]

        self.get_delta()
        self.random_shift()

        self.split_d = 0
        self.num_arrays = 1
        self.fit()

    def get_delta(self):
        """ Get maximum distance in P along one axis """
        axis_dists = np.array([np.max(self.P[:, i]) - np.min(self.P[:, i]) for i in range(self.d)])
        self.delta = np.max(axis_dists)

    def random_shift(self):
        """ Apply a random shift to P so that HST has appropriate distance in expectation """
        # Move pointset to all-positive values
        self.P -= np.min(self.P, axis=1, keepdims=True)

        # Apply a random shift in [0, delta]
        self.P += np.random.random([1, self.d]) * self.delta

    def _update_split_vars(self):
        self.split_d += 1
        if self.split_d >= self.d:
            self.split_d = 0
            self.delta /= 2

    def fit(self):
        while len(self.layers[-1]) < self.n:
            next_layer = []
            next_corners = []
            for i, (points, corner) in enumerate(zip(self.layers[-1], self.corners[-1])):
                left_inds = points[:, self.split_d] <= (corner[self.split_d] + self.delta)
                left_P = points[left_inds]
                right_P = points[np.logical_not(left_inds)]
                if len(left_P) > 0:
                    next_layer.append(left_P)
                    next_corners.append(corner)
                if len(right_P) > 0:
                    next_layer.append(right_P)
                    split_corner = np.copy(corner)
                    split_corner[self.split_d] += self.delta
                    next_corners.append(split_corner)
                self._update_split_vars()
            self.layers.append(next_layer)
            self.corners.append(next_corners)
            print([len(l) for l in self.layers[-1]])


class TreeHST:
    def __init__(self, P, delta=None, split_d=0, prev_split=None, root=True, depth=0):
        """
        Initialize Hierarchically Separated Tree
        input:
            P : numpy array, shape (n x d)
        """
        self.P = P
        self.n, self.d = P.shape
        self.depth = depth
        self.max_depth = 0

        if delta is None:
            self.get_delta()
        else:
            self.delta = delta
        if prev_split is None:
            self.prev_split = np.zeros([self.d])
        else:
            self.prev_split = prev_split

        self.split_d = split_d
        self.root = root # True if this is the root of the tree

        self.weight = None # Weight of the edge pointing to this cell
        self.left_child = None
        self.right_child = None

        if self.root:
            self.random_shift()

        self.fit()
        if self.root:
            self.consolidate_depths()

    def get_delta(self):
        """ Get maximum distance in P along one axis """
        axis_dists = np.array([np.max(self.P[:, i]) - np.min(self.P[:, i]) for i in range(self.d)])
        self.delta = np.max(axis_dists)

    def random_shift(self):
        """ Apply a random shift to P so that HST has appropriate distance in expectation """
        # Move pointset to all-positive values
        self.P -= np.min(self.P, axis=1, keepdims=True)

        # Apply a random shift in [0, delta]
        self.P += np.random.random([1, self.d]) * self.delta

    def _get_split_vars(self):
        next_d = self.split_d + 1
        next_delta = self.delta
        if next_d >= self.d:
            next_d = 0
            next_delta = self.delta / 2

        return next_d, next_delta

    def fit(self):
        if len(self.P) == 1:
            self.max_depth = self.depth
            return

        left_inds = self.P[:, self.split_d] <= (self.prev_split[self.split_d] + self.delta)
        left_P = self.P[left_inds]
        right_P = self.P[np.logical_not(left_inds)]

        next_d, next_delta = self._get_split_vars()

        if len(left_P) >= 1:
            left_split = np.copy(self.prev_split)
            self.left_child = HST(left_P, next_delta, next_d, left_split, root=False, depth=self.depth+1)
            left_depth = self.left_child.max_depth
        else:
            left_depth = 0

        if len(right_P) >= 1:
            right_split = np.copy(self.prev_split)
            right_split[self.split_d] += self.delta
            self.right_child = HST(right_P, next_delta, next_d, right_split, root=False, depth=self.depth+1)
            right_depth = self.right_child.max_depth
        else:
            right_depth = 0

        self.max_depth = max(left_depth, right_depth)

    def consolidate_depths(self, max_depth=None):
        if max_depth is None:
            max_depth = self.max_depth
        # DFS to get to every leaf in the tree that is not at the maximum depth
        if self.left_child is not None and self.left_child.depth < max_depth:
            self.left_child.consolidate_depths(self.max_depth)
        if self.right_child is not None and self.right_child.depth < max_depth:
            self.right_child.consolidate_depths(self.max_depth)

        # If we are at a leaf leaf into ever smaller cells until it is at the correct depth
        # FIXME -- want to do a split here like in the fit function
        print(len(self.P))



if __name__ == '__main__':
    P = np.random.randn(500, 1000)
    k = 10
    eps = 1e-1
    d = np.ceil(np.log(k) / (eps ** 2)).astype(np.int32)
    jl_proj = SparseRandomProjection(d)

    P = jl_proj.fit_transform(P)
    hst = ListHST(P)
    print(len(hst.layers))
