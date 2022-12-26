import numpy as np
from time import time
import matplotlib as plt
from sklearn.random_projection import SparseRandomProjection
import sys
sys.setrecursionlimit(5000)

class HST:
    def __init__(self, points, root=True, depth=0, cell_path=''):
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
        self.root = root # Only true if this is the root of the tree
        self.k_per_target = {}
        self.cost_per_target = {}

        self.parent = None
        self.left_child = None
        self.right_child = None

        if self.root:
            indices = np.expand_dims(np.arange(len(self.points)), -1)
            self.points = np.concatenate((indices, self.points), axis=1)
            self.n, self.d = self.points.shape
            self.d -= 1
            self.max_spread = self.get_spread(spread_func=np.max)
            # FIXME -- do I need to multiply by 2 here for the diam?
            self.diam = self.max_spread * np.sqrt(self.d)
            self.scalar = np.power(0.5, 1.0/(self.d))
            self.random_shift()

    def get_spread(self, spread_func=np.max):
        """ Get maximum distance in points along one axis """
        axis_dists = np.array([np.max(self.points[:, i]) - np.min(self.points[:, i]) for i in range(1, self.d+1)])
        return spread_func(axis_dists)

    def random_shift(self):
        """ Apply a random shift to points so that all points are in the [0, 2 * max_spread]^d box """
        # Move pointset to all-positive values
        self.points[:, 1:] -= np.min(self.points[:, 1:], axis=1, keepdims=True)
        self.points[:, 1:] += 1e-3

        # Apply a random shift in [0, max_spread]
        self.points[:, 1:] += np.random.random([1, self.d]) * self.max_spread

    @property
    def has_left_child(self):
        return self.left_child != None

    @property
    def has_right_child(self):
        return self.right_child != None

    def __len__(self):
        return len(self.points)

def get_split_vars(split_d, max_spread, d):
    next_d = split_d + 1
    next_max_spread = max_spread
    if next_d >= d + 1:
        next_d = 1
        next_max_spread = max_spread / 2

    return next_d, next_max_spread

def propagate_leaf_info(root, max_depth, leaf_diam, top_diam, scalar):
    root.max_depth = max_depth
    root.diam = top_diam * np.power(scalar, root.depth)
    if root.has_left_child or root.has_right_child:
        if root.has_left_child:
            propagate_leaf_info(root.left_child, max_depth, leaf_diam, top_diam, scalar)
        if root.has_right_child:
            propagate_leaf_info(root.right_child, max_depth, leaf_diam, top_diam, scalar)
    else:
        # This is a leaf, so we want to pretend that it is at the max_depth
        # Thus, we need to set its diameter to the diameter of the point at the maximum depth
        root.diam = leaf_diam

def fit_tree(points):
    point_to_cell_dict = {}
    root = HST(points)
    # g_* indicates global variable name for fit_tree()
    #   - this way it won't be confused with variables in _fit()
    g_split_d = 1
    g_max_spread = root.max_spread
    g_prev_split = np.zeros([root.d + 1])

    def _fit(node, split_d, max_spread, prev_split):
        """
        node       -- current node in the HST being processed
        split_d    -- dimension to be split on
        max_spread -- the largest length of the cell along any axis
        prev_split -- the position of the "bottom-left" corner (or high-dim equivalent) of the current cell
        """
        if len(node.points) == 1:
            node.max_depth = node.depth
            point_to_cell_dict[node.points[0, 0]] = node
            return

        left_inds = node.points[:, split_d] <= (prev_split[split_d] + max_spread)
        left_points = node.points[left_inds]
        right_points = node.points[np.logical_not(left_inds)]

        next_d, next_max_spread = get_split_vars(split_d, max_spread, node.points.shape[-1] - 1)

        if len(left_points) >= 1:
            left_split = np.copy(prev_split)
            node.left_child = HST(
                left_points,
                root=False,
                depth=node.depth + 1,
                cell_path=node.cell_path + 'l'
            )
            _fit(node.left_child, next_d, next_max_spread, left_split)
            node.left_child.parent = node
            left_depth = node.left_child.max_depth
        else:
            left_depth = 0

        if len(right_points) >= 1:
            right_split = np.copy(prev_split)
            right_split[split_d] += max_spread
            node.right_child = HST(
                right_points,
                root=False,
                depth=node.depth + 1,
                cell_path=node.cell_path + 'r'
            )
            _fit(node.right_child, next_d, next_max_spread, right_split)
            node.right_child.parent = node
            right_depth = node.right_child.max_depth
        else:
            right_depth = 0

        node.max_depth = max(left_depth, right_depth)

    _fit(root, g_split_d, g_max_spread, g_prev_split)
    # FIXME -- leaf diameter is the diameter at max_depth, so we have to take the geometric ratio constant
    # FIXME    to the power of max_depth and multiply by root.diam
    leaf_diam = root.diam * np.power(root.scalar, root.max_depth)
    propagate_leaf_info(root, root.max_depth, leaf_diam, root.diam, root.scalar)
    return root, point_to_cell_dict

def hst_dist(ptc_dict, a, b, root):
    """
    root -- base of the tree
    a, b -- two points that we want the tree-distance between
    """
    cell_a = ptc_dict[a]
    cell_b = ptc_dict[b]
    lca = ''
    lca_depth = 0
    while cell_a.cell_path[lca_depth] == cell_b.cell_path[lca_depth]:
        lca += cell_a.cell_path[lca_depth]
        lca_depth += 1

    distance = 2 * root.diam * np.sum(np.power(root.scalar, np.arange(lca_depth, root.max_depth)))
    return distance

def get_right_cost(left_cost, cost_list, upper_bound):
    """ Return the smallest v_j such that v_i + v_j > target_cost """
    i = 0
    while left_cost + cost_list[i] < upper_bound:
        i += 1
        if i >= len(cost_list):
            return cost_list[-1]
    return cost_list[i]

def min_k(root, target_cost, cost_list):
    # Check if the dynamic program has already filled this in
    if target_cost in root.k_per_target:
        assert target_cost in root.cost_per_target
        return root.k_per_target[target_cost], root.cost_per_target[target_cost]

    # Base case -- we can afford the entire cell's cost
    if root.diam * len(root) < target_cost:
        return 0, root.diam * len(root)

    # If the cell has left and right subtrees then check all left/right cost splits
    if root.has_left_child and root.has_right_child:
        smallest_k = np.inf
        corresponding_cost = 0
        k_size = 1
        for left_target_cost in cost_list:
            # FIXME -- what do we default k_size to if we break before setting it?
            # if left_target_cost >= target_cost:
            #     break
            right_target_cost = get_right_cost(left_target_cost, cost_list, target_cost)
            # if left_target_cost + right_target_cost >= target_cost:
            #     break

            left_size, left_cost = min_k(root.left_child, left_target_cost, cost_list)
            right_size, right_cost = min_k(root.right_child, right_target_cost, cost_list)
            k_size, k_cost = (left_size + right_size), (left_cost + right_cost)

            # If we could afford each sub-cell but not their sum, need one center for the parent cell
            if k_cost > target_cost and k_size == 0:
                # FIXME -- what is k_cost then??
                k_size = 1
            if k_size < smallest_k:
                smallest_k = k_size
                corresponding_cost = k_cost

        root.k_per_target[target_cost] = k_size
        root.cost_per_target[target_cost] = corresponding_cost
        return k_size, corresponding_cost

    # If just one child, recur on it with no added logic
    if root.has_left_child:
        k_size, corresponding_cost = min_k(root.left_child, target_cost, cost_list)
        root.k_per_target[target_cost] = k_size
        root.cost_per_target[target_cost] = corresponding_cost
        return k_size, corresponding_cost
    if root.has_right_child:
        k_size, corresponding_cost = min_k(root.right_child, target_cost, cost_list)
        root.k_per_target[target_cost] = k_size
        root.cost_per_target[target_cost] = corresponding_cost
        return k_size, corresponding_cost

    # This is a leaf and we could not afford it, so add a center on it
    root.k_per_target[target_cost] = 1
    root.cost_per_target[target_cost] = 0
    return 1, 0

class Center:
    def __init__(self, cell, size):
        self.cell = cell
        self.size = size

def get_cost_list(root, eps):
    min_diam = root.get_spread(spread_func=np.min) * np.sqrt(root.d)
    # Upper bound on the value of delta
    delta = root.diam / min_diam
    base = 1 + eps / (root.d * np.log2(delta))
    min_val = eps * delta / root.n
    max_val = root.n * delta
    cost_list = []
    v = 1
    i = 0
    # FIXME -- the lower bound can be solved for explicitly in faster time...
    while v < max_val:
        if v > min_val:
            cost_list.append(v)
        i += 1
        v = base ** i
    return cost_list

def make_coreset(points, k, eps):
    jl_dim = np.ceil(np.log(k) / (eps ** 2)).astype(np.int32)
    jl_proj = SparseRandomProjection(jl_dim)

    points = jl_proj.fit_transform(points)
    root, ptc_dict = fit_tree(points)
    # true_dist = np.sqrt(np.sum(np.square(points[10] - points[30])))
    # tree_dist = hst_dist(ptc_dict, 10, 30, root)

    cost_list = get_cost_list(root, eps)
    start = time()
    found_k = 0
    min_cost, max_cost = 0, root.n * root.diam
    target_cost = (max_cost + min_cost) / 2
    count = 0
    while found_k != k:
        count += 1
        found_k, found_cost = min_k(root, target_cost, cost_list)
        print(found_k, found_cost, min_cost, target_cost, max_cost)
        if found_k > k:
            min_cost = target_cost
            target_cost = (target_cost + max_cost) / 2
        if found_k < k:
            max_cost = target_cost
            target_cost = (target_cost + min_cost) / 2
        if count == 50:
            break
            
    end = time()
    print(end - start)

if __name__ == '__main__':
    g_points = np.random.randn(2000, 1000)
    g_k = 10
    g_eps = 0.5
    make_coreset(g_points, g_k, g_eps)




# Assume the smallest distance is 1, the largest Delta
# let v_1, ..., v_T be powers of (1+\eps/(d log Delta)) between eps*Delta
# / n and n*Delta
# 
# def SmallestK(node, targetValue):
# #finds the smallest k' necessary to have a k'-median cost <= target
# value in the subtree rooted at node
#    if the number of vertices in the subtree * diameter < targetValue:
#      return 0
#    let L, R the children of node
#    curSmallest= infinity
#    for each v_i, v_j such that v_i + v_j < (1+eps) targetValue:
#      curSmallest min= SmallestK(L, v_i) + SmallestK(R, v_j)
# 
#    return curSmallest
# 
# Note that the inner loop can be implemented in O(T) and not T^2 because
# there is only 1 relevant j for each given i
# When rounding the v_i we need to rescale eps to account for the
# accumulation of error, I think eps/(d log Delta) (ie number of tree
# level) should be the right value but I didn't check the math
