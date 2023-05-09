from time import time
import numpy as np
from tqdm.auto import tqdm
from utils import tree_dist
from hst import hst_dist
from multi_hst import make_multi_HST

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

    left_child = create_sample_tree(left_points, left_inds, sample_tree_ptc_dict)
    right_child = create_sample_tree(right_points, right_inds, sample_tree_ptc_dict)
    node.set_children(left_child, right_child)
    left_child.set_parent(node)
    right_child.set_parent(node)

    return node

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

def set_all_dists(
    sample_tree,
    st_ptc_dict,
    labels,
    curr_node,
    center_node,
    c,
    root,
    hst_ptc_dict,
    norm,
    dist=0,
    depth=0
):
    """
    Given that we have opened a new center, we need to update each point's distance to closest center
    We start from the top unmarked node, since once we've reached a marked node we are no longer closer to its
        already assigned center
    For every leaf corresponding to this top unmarked node, we update the distance based on the closest common ancestor
        of the leaf and the center
    """
    if curr_node.is_leaf:
        curr_point = int(curr_node.points[0][0])
        sample_tree_node = st_ptc_dict[curr_point]

        # Cost update needs to account for point weight since we may be fitting the tree to a coreset
        cost_update = dist * curr_node.weights[0]
        if cost_update < sample_tree_node.cost or sample_tree_node.cost == -1:
            update_sample_tree(sample_tree_node, cost_update)
            labels[curr_point] = c

    for child in curr_node.children:
        new_dist = dist

        # If the center and the current point are in the same subtree, update their distance
        if depth < len(center_node.cell_path) and center_node.cell_path[depth] == child.cell_path[depth]:
            new_dist = tree_dist(root.diam, depth, root.max_depth) ** norm
        if child == center_node:
            new_dist = 0

        set_all_dists(
            sample_tree,
            st_ptc_dict,
            labels,
            child,
            center_node,
            c,
            root,
            hst_ptc_dict,
            norm,
            dist=new_dist,
            depth=depth+1
        )

def multi_tree_open(multi_hst, c, sample_tree, st_ptc_dict, labels, norm):
    for i, (root, hst_ptc_dict) in enumerate(zip(multi_hst.roots, multi_hst.ptc_dicts)):
        leaf = hst_ptc_dict[c]
        top_unmarked = mark_nodes(leaf)
        set_all_dists(sample_tree, st_ptc_dict, labels, top_unmarked, leaf, c, root, hst_ptc_dict, norm)
    return labels

def multi_tree_sample(sample_tree):
    if sample_tree.is_leaf:
        return sample_tree.inds[0]
    left_cost, right_cost = sample_tree.left_child.cost, sample_tree.right_child.cost
    left_prob = left_cost / (left_cost + right_cost)
    if np.random.uniform() < left_prob:
        return multi_tree_sample(sample_tree.left_child)
    return multi_tree_sample(sample_tree.right_child)

def setup_multi_HST(points, norm=2, weights=None, hst_count_from_norm=True, loud=False):
    """
    Value error-checking and building the multi_hst
    This is done in a separate function so that fast_cluster_pp is more readable
    """
    assert norm == 1 or norm == 2
    if loud:
        print('Fitting MultiHST...')
    if weights is None:
        weights = np.ones(len(points))
    if hst_count_from_norm:
        multi_hst = make_multi_HST(points, weights, num_trees=norm+1)
    else:
        multi_hst = make_multi_HST(points, weights, num_trees=1)

    return multi_hst

def fast_cluster_pp(points, k, norm=2, weights=None, hst_count_from_norm=True, allotted_time=np.inf, loud=False):
    start = time()
    multi_hst = setup_multi_HST(points, norm, weights, hst_count_from_norm, loud)
    if time() - start > allotted_time:
        return None, None, None

    centers = []
    n = len(points)
    st_ptc_dict = {i: -1 for i in np.arange(n)}
    sample_tree = create_sample_tree(points, np.arange(n), st_ptc_dict)
    labels = np.ones((n)) * -1
    if loud:
        print('Running Fast-Kmeans++...')
    for i in tqdm(range(k), total=k):
        if len(centers) == 0:
            c = np.random.choice(n)
        else:
            c = multi_tree_sample(sample_tree)
        labels = multi_tree_open(multi_hst, c, sample_tree, st_ptc_dict, labels, norm)
        # FIXME -- I'm 99% sure this can go out of the for-loop
        costs = np.array([st_ptc_dict[i].cost for i in np.arange(n)])
        centers.append(c)
        if time() - start > allotted_time:
            break

    if time() - start > allotted_time:
        print('Ran out of time! Only processed {} of {} centers'.format(len(centers), k))
    return np.array(centers), labels, costs
