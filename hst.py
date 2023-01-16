import numpy as np
import sys
sys.setrecursionlimit(5000)

from utils import tree_dist

### HST code ###
class CubeHST:
    def __init__(self, points, root=True, depth=0, center=None, edge_length=None, cell_path=None):
        """
        Initialize Hierarchically Separated Tree
        input:
            P : numpy array, shape (n x d)
        """
        # FIXME -- There must be a better way to handle the point id's.
        #       -- Appending them outside of the class is so ugly
        self.points = points
        self.depth = depth
        self.max_depth = 0
        self.cell_path = cell_path
        if self.cell_path is None:
            self.cell_path = []
        self.root = root # Only true if this is the root of the tree
        self.marked = False

        self.parent = None
        self.children = []

        if self.root:
            assert (center is None) and (edge_length is None)
            indices = np.expand_dims(np.arange(len(self.points)), -1)
            self.points = np.concatenate((indices, self.points), axis=1)
            self.n, self.d = self.points.shape
            self.d -= 1
            self.edge_length = self.get_spread()
            self.random_shift()
            self.center = np.ones((1, self.d)) * self.edge_length
            self.diam = self.edge_length * np.sqrt(self.d) * 2
        else:
            self.center = center
            self.edge_length = edge_length



    def mark(self):
        if self.marked:
            print(self.is_leaf)
        self.marked = True

    def get_spread(self):
        """ Get maximum distance in points along one axis """
        # FIXME -- this could be faster
        axis_dists = np.array([np.max(self.points[:, i]) - np.min(self.points[:, i]) for i in range(1, self.d+1)])
        return np.max(axis_dists)

    def random_shift(self):
        """ Apply a random shift to points so that all points are in the [0, 2 * edge_length]^d box """
        # Move pointset to all-positive values
        self.points[:, 1:] -= np.min(self.points[:, 1:], axis=0, keepdims=True)

        # Apply a random shift in [0, edge_length]
        shift = np.random.random([1, self.d]) * self.edge_length
        self.points[:, 1:] += shift

        assert np.all(self.points[:, 1:] > 0)
        assert np.all(self.points[:, 1:] < 2 * self.edge_length)

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def __len__(self):
        return len(self.points)

def fit_tree(points):
    point_to_cell_dict = {}
    root = CubeHST(points)

    def _fit(node):
        """
        node       -- current node in the HST being processed
        split_d    -- dimension to be split on
        edge_length -- the largest length of the cell along any axis
        prev_split -- the position of the "bottom-left" corner (or high-dim equivalent) of the current cell
        """
        if len(node.points) == 1:
            node.max_depth = node.depth
            point_to_cell_dict[node.points[0, 0]] = node
            return

        # FIXME -- can probably just look at some of the dimensions -- probably enough to split on anyway
        center_comparison = (node.points[:, 1:] > node.center) * 2 - 1
        # FIXME -- the np.unique does a sort over all n elements. We can instead turn the list of 0s and 1s
        # into a binary number and do list(set()) on those?
        #   - hmm but then we'd have to find the reverse_indices
        dir_from_center, point_inds = np.unique(center_comparison, axis=0, return_inverse=True)
        new_centers = node.center + dir_from_center * node.edge_length / 2
        new_cell_path = node.cell_path + [node]
        for i, new_center in enumerate(new_centers):
            child = CubeHST(
                node.points[np.where(point_inds == i)],
                root=False,
                depth=node.depth + 1,
                center=new_center,
                edge_length=node.edge_length / 2,
                cell_path=new_cell_path
            )
            _fit(child)
            child.parent = node
            node.children.append(child)
            if child.max_depth > node.max_depth:
                node.max_depth = child.max_depth

    _fit(root)
    return root, point_to_cell_dict

def hst_dist(ptc_dict, a, b, root):
    """
    root -- base of the tree
    a, b -- two points that we want the tree-distance between
    """
    cell_a = ptc_dict[a]
    cell_b = ptc_dict[b]
    if cell_a == cell_b:
        return 0
    lca_depth = 0
    while cell_a.cell_path[lca_depth] == cell_b.cell_path[lca_depth]:
        lca_depth += 1
        if lca_depth >= len(cell_a.cell_path) or lca_depth >= len(cell_b.cell_path):
            break
    assert lca_depth > 0

    return tree_dist(root.diam, lca_depth, root.max_depth)

def assert_hst_correctness(root, ptc_dict, points):
    true_dist = np.sqrt(np.sum(np.square(points[10] - points[30])))
    tree_dist = hst_dist(ptc_dict, 10, 30, root)
    assert tree_dist > true_dist
    assert tree_dist < 24 * np.log2(len(root)) * true_dist

