import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import sys
from coreset_sandbox import HST, hst_dist, fit_tree, assert_hst_correctness
sys.setrecursionlimit(5000)

class SampleTree:
    def __init__(self, points, inds, cost=-1):
        self.points = points
        self.inds = inds
        assert len(self.inds) == len(self.points)
        self.cost = cost
        self.parent = None
        self.left_child = None
        self.right_child = None

    def set_children(self, left, right):
        self.left_child = left
        self.right_child = right

    def set_parent(self, parent):
        self.parent = parent

    def __len__(self):
        return len(self.points)

    @property
    def is_leaf(self):
        return len(self) == 1

    @property
    def has_parent(self):
        return self.parent is not None

def create_sample_tree(points, inds, sample_tree_ptc_dict):
    if len(points) == 1:
        assert len(inds) == 1
        node = SampleTree(points, inds)
        sample_tree_ptc_dict[inds[0]] = node
        return node

    node = SampleTree(points, inds)
    split = int(len(points) / 2)
    left_points, left_inds = points[:split], inds[:split]
    right_points, right_inds = points[split:], inds[split:]
    # FIXME remove once sure
    assert len(points) == len(left_points) + len(right_points)

    left_child = create_sample_tree(left_points, left_inds, sample_tree_ptc_dict)
    right_child = create_sample_tree(right_points, right_inds, sample_tree_ptc_dict)
    node.set_children(left_child, right_child)
    left_child.set_parent(node)
    right_child.set_parent(node)

    return node

### HST code ###
class MultiHST:
    def __init__(self, roots, ptc_dicts):
        self.roots = roots
        self.ptc_dicts = ptc_dicts

    def __len__(self):
        return len(self.roots[0])

def multi_tree_dist(multi_hst, a, b):
    min_dist = np.inf
    for i, root in enumerate(multi_hst.roots):
        ptc_dict = multi_hst.ptc_dicts[i]
        dist = hst_dist(ptc_dict, a, b, root)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def assert_multi_hst_correctness(multi_hst, points):
    ### FIXME FIXME -- I think the issue with our HST is that we are splitting along every dimension but still counting
    #                  the diameter at each step. In the Fast-kMeans 3HST example, they don't have a binary tree and so
    #                  they only look at the diameters of cubes, ignoring all the prisms between cubes
    # This may also explain why the HST distances are so huge and why the k-median algorithm isn't working
    true_dist_squared = np.sum(np.square(points[10] - points[30]))
    multi_dist_squared = multi_tree_dist(multi_hst, 10, 30) ** len(multi_hst)
    print(true_dist_squared, (true_dist_squared * int(points.shape[1]) ** 2) / multi_dist_squared)
    assert multi_dist_squared > true_dist_squared
    assert multi_dist_squared < true_dist_squared * int(points.shape[1]) ** 2

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

def mark_nodes(node):
    node.mark()
    if node.parent is None:
        return node
    if node.parent.marked:
        return node
    return mark_nodes(node.parent)

def update_sample_tree(node, cost_update):
    """
        If we are setting a leaf node's cost for the first time, we want to add that amount all the way up the tree.
    If instead we are updating a leaf node's cost, we want to subtract the cost differential from all of its parents.

        If we are setting a non-leaf node's cost for the first time, it must be the case that the leaf node's cost
    was also set for the first time. Thus, we will have received the full cost of that leaf node. Set our cost to that value.
        If we are updating a non-leaf node's cost, then we will have received EITHER the leaf node's first cost (a positive value)
    OR the leaf node's update (a negative value). In either case, we update our cost by this amount.
    """
    # If we have not set this node's cost yet, set it to the current value
    if node.cost == -1:
        node.cost = cost_update
        if node.has_parent:
            # Propagate to parents by adding this cost to their sum
            update_sample_tree(node.parent, cost_update)
    # Otherwise, we have set this node's cost and want to update it with the smaller cost
    else:
        # If this is a leaf node, we
        #   1) set its cost to the smaller value
        #   2) subtract the difference in cost on the leaf from all of its parents' costs
        if node.is_leaf:
            cost_delta = node.cost - cost_update
            node.cost = cost_update
            if node.has_parent:
                # Propagate to parents
                update_sample_tree(node.parent, -1 * cost_delta)
        # Otherwise, this is not a leaf node and we have given it a cost before
        # So just update its current state by the desired change
        else:
            node.cost += cost_update
            if node.has_parent:
                update_sample_tree(node.parent, cost_update)

def set_all_dists(sample_tree, st_ptc_dict, labels, curr_node, c, root, hst_ptc_dict, norm):
    if curr_node.is_leaf:
        curr_point = int(curr_node.points[0][0])
        sq_dist = hst_dist(hst_ptc_dict, c, curr_point, root) ** norm
        sample_tree_node = st_ptc_dict[curr_point]
        if sq_dist < sample_tree_node.cost or sample_tree_node.cost == -1:
            update_sample_tree(sample_tree_node, sq_dist)
            labels[curr_point] = c
    if curr_node.has_left_child:
        set_all_dists(sample_tree, st_ptc_dict, labels, curr_node.left_child, c, root, hst_ptc_dict, norm)
    if curr_node.has_right_child:
        set_all_dists(sample_tree, st_ptc_dict, labels, curr_node.right_child, c, root, hst_ptc_dict, norm)

def multi_tree_open(multi_hst, c, sample_tree, st_ptc_dict, labels, norm):
    for i, (root, hst_ptc_dict) in enumerate(zip(multi_hst.roots, multi_hst.ptc_dicts)):
        leaf = hst_ptc_dict[c]
        top_unmarked = mark_nodes(leaf)
        set_all_dists(sample_tree, st_ptc_dict, labels, top_unmarked, c, root, hst_ptc_dict, norm)
    return labels

def multi_tree_sample(sample_tree):
    if sample_tree.is_leaf:
        return sample_tree.inds[0]
    left_cost, right_cost = sample_tree.left_child.cost, sample_tree.right_child.cost
    left_prob = left_cost / (left_cost + right_cost)
    if np.random.uniform() < left_prob:
        return multi_tree_sample(sample_tree.left_child)
    return multi_tree_sample(sample_tree.right_child)

def fast_cluster_pp(points, k, eps, norm=2):
    assert norm == 1 or norm == 2
    multi_hst = make_multi_HST(points, k, eps, num_trees=norm+1)
    centers = []
    n = len(points)
    # default cost is just some upper bound on the cost, so we do 16d^2 * MaxDist^2
    st_ptc_dict = {i: -1 for i in np.arange(n)}
    sample_tree = create_sample_tree(points, np.arange(n), st_ptc_dict)
    labels = np.ones((n)) * -1
    while len(centers) < k:
        if len(centers) == 0:
            c = np.random.choice(n)
        else:
            c = multi_tree_sample(sample_tree)
        labels = multi_tree_open(multi_hst, c, sample_tree, st_ptc_dict, labels, norm)
        costs = np.array([st_ptc_dict[i].cost for i in np.arange(n)])
        centers.append(c)

    return centers, labels, costs

def evaluate_sensitivities(centers, labels, costs, alpha=10):
    cost_per_center = np.zeros((len(centers)))
    sensitivities = np.zeros((len(labels)))
    for i in range(len(centers)):
        points_in_cluster = np.where(labels == centers[i])
        if len(points_in_cluster) == 0:
            break
        cost_per_center[i] = np.sum(np.take(costs, points_in_cluster))
        sensitivities[points_in_cluster] = costs[points_in_cluster] / cost_per_center[i]
        sensitivities[points_in_cluster] *= alpha
        sensitivities[points_in_cluster] += 1 / len(points_in_cluster[0])
    return sensitivities

def jl_proj(points, k, eps):
    jl_dim = np.ceil(np.log(k) / (eps ** 2)).astype(np.int32)
    jl_model = SparseRandomProjection(jl_dim)
    points = jl_model.fit_transform(points)
    return points

if __name__ == '__main__':
    n_points = 10000
    D = 1000
    num_centers = 10
    alpha = 10
    norm = 1
    g_points, _ = make_blobs(n_points, D, centers=num_centers)
    # g_points = np.random.randn(200, 1000)
    g_k = 10
    g_eps = 0.5
    g_points = jl_proj(g_points, g_k, g_eps)
    d = int(g_points.shape[1])
    centers, labels, costs = fast_cluster_pp(g_points, g_k, g_eps)
    sensitivities = evaluate_sensitivities(centers, labels, costs, alpha=10)

    # FIXME -- Oh ChrIIIiiiissss, what should m be?
    m = int(g_k * d ** norm / g_eps)
    replace = False
    if m > n_points:
        replace = True
    rough_coreset = np.random.choice(n_points, size=m, replace=replace, p=sensitivities/np.sum(sensitivities))
    q_points = g_points[rough_coreset]
    q_labels = labels[rough_coreset]

    # Labels are in range [0, n] but we want them in range [0, k]
    center_to_label_dict = {centers[i]: i for i in range(len(centers))}
    labels = np.array([center_to_label_dict[c] for c in labels])

    embedding = PCA(n_components=2).fit_transform(q_points)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=q_labels)
    plt.show()
