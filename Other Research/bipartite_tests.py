from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.colors
import random
import numpy as np
import networkx as nx
from networkx import bipartite
import itertools
import math
import time
import os

data,labels = make_blobs(n_samples = 1500, random_state = 11037)
m = 50

t1 = time.time()
best, cost = None, np.inf
sample = data[0:50]
datag = data[50:]
B = nx.DiGraph()
B.add_nodes_from( [tuple(datag[i]) for i in range(len(datag))], bipartite = 0, demand = -1)
B.add_nodes_from( [tuple(sample[i]) for i in range(len(sample))], bipartite = 1, demand = len(datag)/m,  capacity = len(datag)/m)

t2 = time.time()
print('Graph created in {}s'.format(t2-t1))

top_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
bottom_nodes = set(B) - top_nodes

dist = lambda x, y: np.sqrt(sum([x[i]**2 + y[i]**2 for i in range(len(x))]))

for node in bottom_nodes:
   
    B.add_weighted_edges_from( [(node, pt, round(dist(node, pt * 100))) for pt in top_nodes], 'weight')
    
t3 = time.time()

print('Edges added in {}s'.format(t3-t2))

cost, mfwm = nx.capacity_scaling(B, 'demand', 'capacity', 'weight')#, top_nodes[0], bottom_nodes[0])
# mfwm =  bipartite.maximum_matching(B, top_nodes)#, 'weight')

t4 = time.time()
print('Matching computed in {}s'.format(t4-t3))

print(mfwm)
# nx.draw(B)
# plt.show()

nr = math.factorial(len(data)) / (math.factorial(len(data) - m) * math.factorial(m))

for i, sample in enumerate(itertools.combinations(data, m)):
    print('Running iteration {}/{}...'.format(i+1,nr))
    datag = data[[bool((i in np.array(sample))^1) for i in data]]
    dist = lambda x, y: np.sqrt(sum([x[i]**2 + y[i]**2 for i in range(len(x))]))

    B = nx.DiGraph()
    dmd_s = len(datag)/m
    dmd_p = dmd_s * len(sample) / len(data)

    B.add_nodes_from( [tuple(datag[i]) for i in range(len(datag))], bipartite = 0, demand = dmd_p)
    B.add_nodes_from( [tuple(sample[i]) for i in range(len(sample))], bipartite = 1, demand = dmd_s,  capacity = len(datag)/(k))

    top_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
    bottom_nodes = set(B) - top_nodes


    for node in bottom_nodes:
       
        B.add_weighted_edges_from( [(node, pt, round(dist(node, pt * 100))) for pt in top_nodes], 'weight')
        

    try:
        cost, mfwm = nx.capacity_scaling(B, 'demand', 'capacity', 'weight')#, top_nodes[0], bottom_nodes[0])
    except:
        cost = np.inf
        mfwm = None
        
    if scost > cost:
        best = sample
        scost = cost
    # print("\014")
print(best, scost)
