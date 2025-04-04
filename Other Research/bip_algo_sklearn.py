# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:47:54 2025

@author: 20190819
"""
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

#todo: better initialization
# def kmeanspp(data, k):
#     out = []
#     data2 = data.copy()
#     c1 = random.choice(data)
#     out.append(c1)
#     data2 = np.array([i for i in list(data) if all(i != np.array(c1))])#remove centers already chosen
#     c_next = None
#     while len(out) < k:
#         probs = []
#         for p in data2:
#             num = min([dist(p, c) for c in out])
#             den = sum([min([dist(q, c) for c in out]) for q in data2])# if all(q != p)])
#             probs.append(num/den)        
#         choice = np.random.choice(range(data2.shape[0]), p = probs)
#         c_next = data2[choice]
#         out.append(c_next)
#         data2 = np.array([i for i in list(data) if all(i != np.array(c_next))])#remove c_next
#     return out, probs

#variables
#sample weight  = n/m
#enumerate every k of SAMPLE POINT
#then we do bipartite with SAMPLE POINTS and k centers chosen FROM SAMPLE
#but edge weight is dist is from every sample point to every center
#demand <- n/m (nr of sample pts)
#max capacity <- n/k (nr of clusters)
alpha = 0.4 #unused
cent =  5
data,labels = make_blobs(n_samples = 150000, centers= cent)#, centers = [[-5, -5], [0,0], [5,5]])
m = 30000
k = 15

#complex variables
sample = data[random.sample(range(data.shape[0]), k= m+k)]
w_s = int(len(data)/m)
demand = int(len(data)/m)
capacity = int(len(data)/k)

#implement kmeans here
centers, labels = sklearn.cluster.kmeans_plusplus(sample, k)
for c in centers:
    sample = np.array([i for i in list(sample) if all(i != np.array(c))])

#sanity checks
assert len(sample) == m
assert len(centers) == k
#graph creation
B = nx.DiGraph()

B.add_nodes_from( [tuple(centers[i]) for i in range(len(centers))], bipartite = 0, demand = capacity,  capacity = capacity)
B.add_nodes_from( [tuple(sample[i]) for i in range(len(sample))], bipartite = 1, demand = -demand, weight = w_s)
top_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}

bottom_nodes = set(B) - top_nodes


for node in bottom_nodes:
   
    B.add_weighted_edges_from( [(node, pt, round(dist(node, pt) * 100)) for pt in top_nodes], 'weight')
    
cost, mfwm = nx.capacity_scaling(B, 'demand', 'capacity', 'weight')#, top_nodes[0], bottom_nodes[0])
# matching = bipartite.minimum_weight_full_matching(B, top_nodes, 'weight')
# print(matching.keys())
get_max_single = lambda lol: list(lol.keys())[list(lol.values()).index(max(lol.values()))]
    
def get_max(val):
    maxval = get_max_single(val)
    maxes = [i for i in val if i == maxval]
    if len(maxes) > 1:
        print(maxes)
        return 0
    else:
        out = maxes[0]
    
    return out

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
matches = make_matches(centers, mfwm)
arrayscatter = lambda x, c, l=False: plt.scatter(np.array(x)[:,0], np.array(x)[:,1], c=c, label=l)
cntr = 0
clrmap = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink', 'purple', 'olive', 'gray', 'brown',
          'aquamarine', 'mediumseagreen', 'xkcd:sky blue', 'xkcd:eggshell', 'xkcd:ochre', 'xkcd:silver',
          'xkcd:maize', 'xkcd:fawn', 'xkcd:banana', 'xkcd:pale cyan']

for ky in matches.keys():
    tplt = [i for i in sample if (tuple(i) in matches[ky])]
    arrayscatter(tplt, clrmap[cntr])
    cntr += 1
arrayscatter(centers, c='white', l=True)
# plt.legend(['Set 1', 'Set 2', 'Set 3', 'C1', 'C2', 'C3'])

path = '''C:\\Users\\20190819\\Desktop\\Master Thesis Stuff\\plots\\'''
plt.savefig(path+'k{}_m{}_{}_nonoise.png'.format(k, m, cent))
plt.show()
#no need to nCk, just run kmeans++
#pick one sample pt as 1st center, pick one with kmeans++ algo up to k
#k centers obtained :) 

#another version to try: left side is all points with demand 1
#see if reasonably balanced

# sample = data[0:50]
# datag = data[50:]
