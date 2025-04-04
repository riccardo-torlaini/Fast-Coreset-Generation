
    
            
from sklearn.datasets import make_blobs
import sklearn.cluster
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import random
import numpy as np
import networkx as nx
from networkx import bipartite
import itertools
import math
import time
import os

#functions
dist = lambda x, y: np.sqrt(sum([(x[i] - y[i])**2 for i in range(len(x))]))

get_max = lambda lol: list(lol.keys())[list(lol.values()).index(max(lol.values()))]
arrayscatter = lambda x, c, l=False: plt.scatter(np.array(x)[:,0], np.array(x)[:,1], c=c, label=l, s=1)

def make_matches(centers, mfwm):
    out = {}
    for center in centers:
        out[tuple(center)] = []
    for key in mfwm.keys():
        current = mfwm[key]
        if current != {}: #actual point
            ctr = get_max(current)
            out[ctr].append(key)
        else: #center
            pass
        
    return out

#simple variables

alpha = 0.3
n= 3000
k = 3

data2,labels = make_blobs(n_samples= n+k, centers = 3)
m = round((k*np.log(k))/alpha)
y = int((k/alpha)**k)

#complex variables



# Suppose we want to solve balanced k-means where every cluster should be of size a >=alpha n/k. 
# We consider a counter y=(k/\alpha)^k. 
# For i=1 to y we do the following: 
    # We sample a set K of k points uniformly at random. 
    # Then, we build a graph where on the right side we have all points in P and on the left side we have K. 
    # The edge weights between the left and right points is the distance. 
    # Now, you run max-flow min-cut. For every run i, you compute the cost z_i. 
    # At the end of this y iteration, you report the cost whose z_i is minimum.
costs = []
mincost = np.inf 
graph = None
centers = None
for i in range(y):
    print('Running iteration {}'.format(i))
    data=data2.copy()

    sample = data[random.sample(range(data.shape[0]), k= k)]
    for c in sample:
        data = np.array([i for i in list(data) if all(i != np.array(c))])

    w_s = int(len(data)/m)
    demand = int(len(data)/m)
    capacity = int(len(data)/k)    
    
    B = nx.DiGraph()
    
    B.add_nodes_from( [tuple(sample[i]) for i in range(len(sample))], bipartite = 0, demand = capacity,  capacity = capacity)
    B.add_nodes_from( [tuple(data[i]) for i in range(len(data))], bipartite = 1, demand = -1, weight = 1)
    top_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
    bottom_nodes = set(B) - top_nodes
    
    for node in bottom_nodes:
       
        B.add_weighted_edges_from( [(node, pt, round(dist(node, pt)*100)) for pt in top_nodes], 'weight')

    cost, mfwm = nx.capacity_scaling(B, 'demand', 'capacity', 'weight')#, top_nodes[0], bottom_nodes[0])
    costs.append((i,cost))
    if cost < mincost:
        mincost = cost
        graph = mfwm
        centers = sample
  

matches = make_matches(centers, graph)
arrayscatter = lambda x, c, l=False: plt.scatter(np.array(x)[:,0], np.array(x)[:,1], c=c, label=l, s=1)
clrmap = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink', 'purple', 'olive', 'gray', 'brown',
          'aquamarine', 'mediumseagreen', 'xkcd:sky blue', 'xkcd:eggshell', 'xkcd:ochre', 'xkcd:silver',
          'xkcd:maize', 'xkcd:fawn', 'xkcd:banana', 'xkcd:pale cyan']

for i, ky in enumerate(matches.keys()):
    tplt = list(matches.values())[i]
    arrayscatter(tplt, clrmap[i])
arrayscatter(centers, c='white', l=True)
path = '''C:\\Users\\20190819\\Desktop\\Master Thesis Stuff\\plots\\'''

plt.savefig(path+'iterative_k{}_alpha{}_n{}.png'.format(k, alpha, n))

# plt.legend(['Set 1', 'Set 2', 'Set 3', 'C1', 'C2', 'C3'])
plt.show()