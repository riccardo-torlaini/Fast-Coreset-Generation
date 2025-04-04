#independent research: develop one that's running but with 'gradient descent'
#centers sampled with probability that increases if cost is decreased by using them
# -> initial prob of all pts is n/k, increase the sampled ones by x and decrease all others s.t. it all adds up to 1
#shouldn't get stuck since prob. based? -> temperature, doesn't get stuck in local optimum
#might have better performance esp for centers selection but is kinda putting the cart before the horses
#eh something to show morteza ig


    
            
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

alpha = 0.4
beta = 0.01
gamma = 0.5
n= 7200
k = 3

data2,labels = make_blobs(n_samples= n+k, centers = 3)
m = round((k*np.log(k))/alpha)
y = int((k/alpha)**k)
costs = []
mincost = np.inf 
graph = None
centers = None

eps = 1/(n+k)
probs = [eps for i in range(n+k)]

x1, x2 = None, None

for i in range(y):
    print('Running iteration {}'.format(i))
    data=data2.copy()

    sample = data2[np.random.choice(range(data.shape[0]), k, p = probs, replace = False)]
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
    
    
    cpts = [ [p for p in data2 if dist(p, c) < gamma] for c in sample]
    closepts = []
    [closepts.extend(i) for i in cpts]
    lenx = len([str(cpt) for cpt in closepts])
    leny = n+k-lenx
    if cost < mincost:
        mincost = cost
        graph = mfwm
        centers = sample
        
        #update probs
        #warning: dark python magic here. 
        
        x1 = (1+lenx/(n+k))*x1 if x1 else 2*eps
        x2 = max((1-(x1)*lenx)/leny, 0)
        
        probs = [x1 if (str(data2[i]) in [str(cpt) for cpt in closepts]) else x2 for i in range(n+k)]
   
    # else:
    #     x1 = 0.55*x1 if x1 else 0.55*eps
    #     x2 = (1-(x1)*lenx)/leny
    #     probs = [x1 if (str(data2[i]) in [str(cpt) for cpt in closepts]) else x2 for i in range(n+k)]


matches = make_matches(centers, graph)
arrayscatter = lambda x, c, l=False, s=1: plt.scatter(np.array(x)[:,0], np.array(x)[:,1], c=c, label=l, s=s)
clrmap = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink', 'purple', 'olive', 'gray', 'brown',
          'aquamarine', 'mediumseagreen', 'xkcd:sky blue', 'xkcd:eggshell', 'xkcd:ochre', 'xkcd:silver',
          'xkcd:maize', 'xkcd:fawn', 'xkcd:banana', 'xkcd:pale cyan']

for i, ky in enumerate(matches.keys()):
    tplt = list(matches.values())[i]
    arrayscatter(tplt, clrmap[i])
arrayscatter(centers, c='white', l=True)

path = '''C:\\Users\\20190819\\Desktop\\Master Thesis Stuff\\plots\\'''

plt.savefig(path+'improved_iterative_n{}_k{}_centers{}_y{}_gamma{}.png'.format(n, k, len(sample), y, gamma))
# plt.legend(['Set 1', 'Set 2', 'Set 3', 'C1', 'C2', 'C3'])
plt.show()