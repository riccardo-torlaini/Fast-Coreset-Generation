# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 23:09:46 2025

@author: 20190819
"""
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.colors

data,labels = make_blobs(n_samples = [70, 70, 10], n_features =3, centers = [[-6, -6], [-1, -6], [5,5]], random_state = 11037)

topltx = [x[0] for x in data]
toplty = [x[1] for x in data]

colorplt0 = [0 for point in data]
colorplt1 = [1 if i < 0 else 0 for i in toplty]
colorplt2 = []
cmapsw = []
for i in data:
    x, y = i[0], i[1]
    if y > 0:
        colorplt2.append(2)
        cmapsw.append('gray')
    elif x > -3:
        colorplt2.append(1)
        cmapsw.append('red')
    else:
        colorplt2.append(0)
        cmapsw.append('blue')
        

plt.scatter(topltx, toplty, marker=(6, 0, 0), s=10, c=colorplt2, cmap = 'Set1')

# n_samples = 50000
# n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here

# Generate 3 blobs with 2 classes where the second blob contains
# half positive samples and half negative samples. Probability in this
# blob is therefore 0.5.
# centers = [(-5, -5), (0, 0), (5, 5)]
# X, y = make_blobs(n_samples=n_samples, centers=centers, shuffle=False, random_state=42)
# plt.scatter(X, y)