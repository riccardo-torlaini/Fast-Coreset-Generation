from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.colors
import random
import numpy as np

data,labels = make_blobs(n_samples = 15000, random_state = 11037)
array_in = lambda x, y: (x == y).all(axis=1).any()
euc_dist = lambda pt, ctr: np.sqrt(np.sum([(pt[i] - ctr[i])**2 for i in range(len(ctr))]))

def init_clust(data, n):
    out = []
    try:
        out = random.sample(sorted(data), n)
    except ValueError:
        print(data)
        out = data
    return out
def cluster(data, n):
    initcenters = init_clust(data, n)
    y = [[], [], []]
    for pt in data:
        dists = [euc_dist(pt, ctr) for ctr in initcenters]
        group = dists.index(min(dists))
        y[group].append(pt)
            
    return y
def point_split(data, sf):
    out = []
    points = list(data.copy())
    while len(points) != 0:
        if len(points) < sf:
            out.append([points])
            points = []
        else:
             toadd = random.sample(points, sf)
             out.append(toadd)
             points = [p for p in points if array_in(p, toadd) == False]
    return out

def distributed_clustering(data, gamma):
    # 1. PARTITION points arbitrarily into 2n^[(1+gamma)/2] sets
    split_factor = round(2 * data.shape[0] ** ((1+gamma)/2))
    splits = point_split(data, split_factor)
    # 2. COMPUTE composable 2^P mapping coreset on each machine in parallel -> 
    # obtain f, multisets S1, ..., S_(2n^[(1+gamma)/2]), each with roughly delta-O(k) distinct pts
    multisets = []
    for df in splits:
        multisets.append(cluster(df[0], 3))
    print(multisets)
    # 3. PARTITION computed coreset again into n^(1/4) sets
    # splits_2 = 
    # for mset in multisets:
        
    # 4. COMPUTE composable 2^P mapping coreset on each machine in parallel -> 
    # obtain f', multisets S'1, ..., S'_(n^(1/4)), each with delta-O(k) distinct pts
    
    # 5. MERGE all S'1, ..., S'_(n^(1/4)) on single machine, 
    # compute clustering using sequential space-efficient alpha-approx algorithm
    
    # 6. MAP BACK points in S'1, ..., S'_(n^(1/4)) to points in S1, ..., S_(2n^[(1+gamma)/2]) 
    #with function f'^(-1) -> obtain clustering of pts in S1, ..., S_(2n^[(1+gamma)/2])
    
    # 7. MAP BACK points in S1, ..., S_(2n^[(1+gamma)/2]) to points in V with f^(-1) -> 
    # obtain clustering of INITIAL POINTSET

        
    
    
    
distributed_clustering(data, 0.5)    