# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 01:07:01 2025

@author: 20190819
"""
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

#functions
dist = lambda x, y: np.sqrt(sum([x[i]**2 + y[i]**2 for i in range(len(x))]))

def kmeanspp(data, k):
    out = []
    data2 = data.copy()
    c1 = random.choice(data)
    out.append(c1)
    data2 = np.array([i for i in list(data) if all(i != np.array(c1))])#remove centers already chosen
    c_next = None
    while len(out) < k:
        probs = []
        for p in data2:
            num = min([dist(p, c) for c in out])
            den = sum([min([dist(q, c) for c in out]) for q in data2])# if all(q != p)])
            probs.append(num/den)        
        choice = np.random.choice(range(data2.shape[0]), p = probs)
        c_next = data2[choice]
        out.append(c_next)
        data2 = np.array([i for i in list(data) if all(i != np.array(c_next))])#remove c_next
    return out

#variables
#sample weight  = n/m
#enumerate every k of SAMPLE POINT
#then we do bipartite with SAMPLE POINTS and k centers chosen FROM SAMPLE
#but edge weight is dist is from every sample point to every center
#demand <- n/m (nr of sample pts)
#max capacity <- n/k (nr of clusters)
alpha = 0.4 #unused

costs = []
m, k = 10000, 5
random_states = [1, 0, 1000000, 11037, 420, 12345, 9876, 3141, 2137, 404]
for i in random_states:
    
    data,labels = make_blobs(n_samples = 150000, random_state=i)

    #complex variables
    sample = data[random.sample(range(data.shape[0]), k= m + k)]
    w_s = int(len(data)/m)
    demand = int(len(data)/m)
    capacity = int(len(data)/k)
    
    #implement kmeans here
    centers = kmeanspp(sample, k)
    for c in centers:
        sample = np.array([i for i in list(sample) if all(i != np.array(c))])
        
    assert len(sample) == m
    assert len(centers) == k
    #graph creation
    B = nx.DiGraph()
    
    B.add_nodes_from( [tuple(sample[i]) for i in range(len(sample))], bipartite = 1, demand = -demand, weight = w_s)
    B.add_nodes_from( [tuple(centers[i]) for i in range(len(centers))], bipartite = 0, demand = capacity,  capacity = capacity)
    
    # print(sum([i['demand'] for i in B.nodes(data=True)]))
    top_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
    bottom_nodes = set(B) - top_nodes
    
    
    for node in bottom_nodes:
       
        B.add_weighted_edges_from( [(node, pt, round(dist(node, pt * 100))) for pt in top_nodes], 'weight')
        
    cost, mfwm = nx.capacity_scaling(B, 'demand', 'capacity', 'weight')#, top_nodes[0], bottom_nodes[0])
    # print(cost)
    costs.append(cost)
    #no need to nCk, just run kmeans++
    #pick one sample pt as 1st center, pick one with kmeans++ algo up to k
    #k centers obtained :) 
    
    #another version to try: left side is all points with demand 1
    #see if reasonably balanced
    
    # sample = data[0:50]
    # datag = data[50:]
for i in range(len(costs)):
    print('Iteration i: cost = {}, data generation seed was {}'.format(i, costs[i], random_states[i]))