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

#basic variables
alpha = 0.4 #unused
data,labels = make_blobs(n_samples = 150000, centers= 30)
m = 2000
m2 = 50
k = 5

morepts = np.array([(29+(0.03*i), 29+(0.015*i)) for i in range(0, m2)])


#complex variables
sample = data[random.sample(range(data.shape[0]), k= m-m2+k)]
w_s = int(len(data)/m)
demand = int(len(data)/m)
capacity = int(len(data)/k)

#implement kmeans here
centers, labels = sklearn.cluster.kmeans_plusplus(sample, k)
for c in centers:
    sample = np.array([i for i in list(sample) if all(i != np.array(c))])

#adding noise
reshapelen = int((len(sample)+len(morepts)))
sample = np.append(sample, morepts).reshape(reshapelen, 2)

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
   
    B.add_weighted_edges_from( [(node, pt, round(dist(node, pt)* 100)) for pt in top_nodes], 'weight')
    
cost, mfwm = nx.capacity_scaling(B, 'demand', 'capacity', 'weight')#, top_nodes[0], bottom_nodes[0])
# matching = bipartite.minimum_weight_full_matching(B, top_nodes, 'weight')
# print(matching.keys())
get_max = lambda lol: list(lol.keys())[list(lol.values()).index(max(lol.values()))]
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

plt.figure()
for i,key in enumerate(matches.keys()):
    arrayscatter(list(matches[key]), c=clrmap[i])
arrayscatter(centers, c='white', l=True)

path = '''C:\\Users\\20190819\\Desktop\\Master Thesis Stuff\\plots\\'''
plt.savefig(path+'k{}_m{}_{}_{}noise.png'.format(k, m, len(centers), len(morepts)))
plt.show()
print(cost, [len(m) for m in matches.values()])


#no need to nCk, just run kmeans++
#pick one sample pt as 1st center, pick one with kmeans++ algo up to k
#k centers obtained :) 

#another version to try: left side is all points with demand 1
#see if reasonably balanced

# sample = data[0:50]
# datag = data[50:]
