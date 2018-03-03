#File for general DM stuff

import os

import numpy as np
import pandas as pd
import copy
from statistics import mode

import sklearn
from sklearn.cluster import KMeans
from sklearn.mixture import gaussian_mixture
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


file = os.sep.join(['seeds_dataset.txt'])
data = pd.read_csv(file, sep="\t", header=None, names=["Area", "Perimeter", "Compactness", "Length of Kernel","Width of Kernel","Asymmetry coefficient"," Length of kernel groove","h"])

dataVals = data.values

data_classes = []
for instance in dataVals:
    data_classes.append(instance[-1])

data=data.drop(columns=['h'])
data.head()

dataVals = data.values
#print(dataVals)



finalA = []
finalB = []
X = []
for instance in dataVals:
    tempL = []
    tempL.append(instance[0])
    tempL.append(instance[1])
    tempL.append(instance[2])
    tempL.append(instance[3])
    tempL.append(instance[4])
    tempL.append(instance[5])
    tempL.append(instance[6])
    X.append(tempL)
#print(finalA)
#print(finalB)

X = np.array(X)
#print(X)





# Initialize the clusterer with 3 value and a random generator
# seed of 10 for reproducibility.
clusterer = gaussian_mixture.GaussianMixture(n_components=3)
clusterer.fit(X)
cluster_labels = clusterer.predict(X)

#k means error reporting
#find the most popular per 70
#print(type(cluster_labels))
cluster_labels_list = cluster_labels.tolist()
#print(type(cluster_labels_list))

mode1 = mode(cluster_labels_list[:70])
mode2 = mode(cluster_labels_list[70:140])
mode3 = mode(cluster_labels_list[140:])

#print(mode1,mode2,mode3)

Ttrue = 0
Tfalse = 0
Ttrue += cluster_labels_list[:70].count(mode1)
Tfalse += 70-cluster_labels_list[:70].count(mode1)
Ttrue += cluster_labels_list[70:140].count(mode2)
Tfalse += 70-cluster_labels_list[70:140].count(mode2)
Ttrue += cluster_labels_list[140:].count(mode3)
Tfalse += 70-cluster_labels_list[140:].count(mode3)

"""
print(cluster_labels_list[:70].count(mode1))
print(70-cluster_labels_list[:70].count(mode1))
print(cluster_labels_list[70:140].count(mode2))
print(70-cluster_labels_list[70:140].count(mode2))
print( cluster_labels_list[140:].count(mode3))
print(70-cluster_labels_list[140:].count(mode3))
"""

print(str(Ttrue) + "/" + str(Ttrue + Tfalse))
print("Accuracy: " + str(Ttrue / (Ttrue + Tfalse)))

print(cluster_labels_list)