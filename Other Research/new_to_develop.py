# test case: 3 blobs, centers lead to overlapping clusters + faraway points, as in picture 
# 2 balls are 10k, 1 is 5k (or one is also 10k)
#come up with more test cases
# make visualization, also print cluster size

#also try different algos (benchmark): 
    #repeat x times: 
        #sample k points uniformly at random from original data
        #construct graph, right side centers, left side all points
        #demand of left side is 1, capacity of centers is same as before (n/k)
        #do maxcutminflow
        #get cost
        #next iteration, again sample k pts and do as above...
        #keep track of best cost 
        #recurse only if new cost is < at least b*best_cost
     
        

        
#run 3 algorithms on different datasets, visualize and print relevant data (cost, cluster sizes, etc)


     
    
        
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

#variables
#sample weight  = n/m
#enumerate every k of SAMPLE POINT
#then we do bipartite with SAMPLE POINTS and k centers chosen FROM SAMPLE
#but edge weight is dist is from every sample point to every center
#demand <- n/m (nr of sample pts)
#max capacity <- n/k (nr of clusters)
alpha = 0.4 #
n= 10000
data2,labels = make_blobs(n_samples= n )
data=data2.copy()
k = 5
m = round((k*np.log(k))/alpha)

#complex variables
sample = data[random.sample(range(data.shape[0]), k= m)]

for c in sample:
    data = np.array([i for i in list(data) if all(i != np.array(c))])

w_s = int(len(data)/m)
demand = int(len(data)/m)
capacity = int(len(data)/k)    

#hierarchical balanced clustering
    #sample S points, S = [k*log(k)]/alpha
    #left side <- sample pts, right side <- original pts    
    #original pts <- demand 1, sample pts <- capacity n/|S|


    #maxflowmincut on this
    #store assignment
    #now with S points, do ALL pairwise distances, get min dist
    #choose one of min dist pts as center, assing other pt to it w/ associated pts as above
    #treat s1 and s2 as same center (kind of hierarchical clustering)
    #for any s3 in S, now I have dist(s3, s1 AND s2) computed earlier
    #now if:
        # min_dist is between some s3, s4 -> merge those
        # min_dist is between some s3 and agglomerated point: add s3 to that point (i.e. s1 AND s2 AND s3)
    #if at any point a cluster has at least n/k pts, that cluster is FIXED (balanced), so we ignore it
    #continue till we end up with k points
B = nx.DiGraph()

B.add_nodes_from( [tuple(sample[i]) for i in range(len(sample))], bipartite = 0, demand = demand,  capacity = capacity)
B.add_nodes_from( [tuple(data[i]) for i in range(len(data))], bipartite = 1, demand = -1, weight = 1)
top_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}

bottom_nodes = set(B) - top_nodes


for node in bottom_nodes:
   
    B.add_weighted_edges_from( [(node, pt, round(dist(node, pt)* 100)) for pt in top_nodes], 'weight')

print('Graph created at time: {}'.format(time.time()))
    
cost, mfwm = nx.capacity_scaling(B, 'demand', 'capacity', 'weight')#, top_nodes[0], bottom_nodes[0])
print('Min-flox max-cut computed at time: {}'.format(time.time()))


matches = make_matches(sample, mfwm)

print('Matches dictionary created at time: {}'.format(time.time()))

def mindist(d):
    keys = d.keys()
    pts = None
    mindistance = np.inf
    for key in itertools.combinations(keys, 2):
        key0 = key[0]
        key1 = key[1]
        curdist = 0
        
        try:
        #best way I could manage to check if this is a tuple or a tuple of tuples
            mask1 = all( [ all( [type(key0[f][g]) == tuple for g in range(2)] ) for f in range(2)] )
            mask2 = all( [ all( [type(key1[f][g]) == tuple for g in range(2)] ) for f in range(2)] )
            
            if mask1 and mask2:
                
                for r in key0:
                    for j in key1:
                        #r, j -> tuples of tuples ((0,1), (1,2))
                        for r1 in r:
                            for j1 in j:
                                
                                curdist = dist(r1, j1)
                                if curdist < mindistance:
                                    mindistance = curdist
                                    pts = key0, key1
            if mask1:
                for r in key0:
                    for j in key1:
                        for r1 in r:
                            
                            curdist = dist(r1, j)
                            if curdist < mindistance:
                                mindistance = curdist
                                pts = key0, key1
            if mask2:
                for r in key0:
                    for j in key1:
                        #r, j -> tuples of tuples ((0,1), (1,2))
                        for j1 in j:
                           
                            curdist = dist(r1, j1)
                            if curdist < mindistance:
                                mindistance = curdist
                                pts = key0, key1
                                
        except IndexError:
            
            mask3 = all( [type(h) == tuple for h in key0] )
            mask4 = all( [type(h) == tuple for h in key1] )
            
            if mask3 and mask4:
                print('This is a stange place. What happened here?')
                print(key0, mask1, mask3)
                print(key1, mask2, mask4)
                break 
            elif mask3:
                curdist = min( [ dist(e, key1) for e in key0] )
                if curdist < mindistance:
                    mindistance = curdist
                    pts = key0, key1
            elif mask4:
                print(key0, key1, mask3, mask4)
                curdist = min( [ dist(key0, e) for e in key1] )
                if curdist < mindistance:
                    mindistance = curdist
                    pts = key0, key1
            
    return pts

def tupleadd(pt1, pt2):
    mask1 = all([type(p) == tuple for p in pt1])
    mask2 = all([type(p2) == tuple for p2 in pt2])
    
    if mask1 and mask2:
        return pt1 + pt2
    elif mask1:
        return pt1 + (pt2,)
    elif mask2:
        return (pt1,) + pt2
    else:
        return (pt1,)+(pt2,)
    

hrc_mtch = matches.copy()
out = {}
while len(out) != k and len(hrc_mtch) > 1:
    while not any([len(i) >= int(n/k) for i in hrc_mtch.values()]):
        if all([type(t[0]) == np.float64 for t in hrc_mtch.keys()]):
            mindstpts = min([((p[0], p[1]), dist(p[0], p[1])) for p in itertools.combinations(hrc_mtch.keys(), 2)], key = lambda x: x[1])[0]
            
            pt1, pt2 = mindstpts
            assert len(pt1) == 2
            assert len(pt2) == 2
            
            newkey = tupleadd(pt1, pt2)
        else:
            pt1, pt2 = mindist(hrc_mtch)
            newkey = tupleadd(pt1, pt2)
        pt1 = tuple(pt1)
        pt2 = tuple(pt2)    
        hrc_mtch[pt1].extend(hrc_mtch[pt2])
        hrc_mtch[tuple(newkey)] = hrc_mtch[pt1]
    
        del hrc_mtch[pt1]
        del hrc_mtch[pt2]
    else:
        keyind = [len(i) >= int(n/k) for i in hrc_mtch.values()].index(True)
        keyact = list(hrc_mtch.keys())[keyind]
        out[keyact] = hrc_mtch[keyact]
        del hrc_mtch[keyact]
else:
    if len(hrc_mtch) == 1:
        out[list(hrc_mtch.keys())[0]] = list(hrc_mtch.values())
        

print('Code finished running at time: {}\n With resulting assignment:'.format(time.time()))

# matches = make_matches(centers, mfwm)
arrayscatter = lambda x, c, l=False: plt.scatter(np.array(x)[:,0], np.array(x)[:,1], c=c, label=l)
cntr = 0
clrmap = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink', 'purple', 'olive', 'gray', 'brown',
          'aquamarine', 'mediumseagreen', 'xkcd:sky blue', 'xkcd:eggshell', 'xkcd:ochre', 'xkcd:silver',
          'xkcd:maize', 'xkcd:fawn', 'xkcd:banana', 'xkcd:pale cyan']

for ky in out.keys():
    tplt = [i for i in out[ky]]
    arrayscatter(tplt, clrmap[cntr])
    cntr += 1
# arrayscatter(centers, c='white', l=True)
# plt.legend(['Set 1', 'Set 2', 'Set 3', 'C1', 'C2', 'C3'])

path = '''C:\\Users\\20190819\\Desktop\\Master Thesis Stuff\\plots\\'''
plt.savefig(path+'hierarchical_k{}_m{}.png'.format(k, m))
plt.show()

print(out.keys(), [len(i) for i in out.values()])