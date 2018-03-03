import os

import numpy as np
import pandas as pd
import copy
from statistics import mode

import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

import tensorflow as tf
import numpy as np



def input_fn():
    return tf.constant(data.as_matrix(), tf.float32, data.shape), None


def ScatterPlot(X, Y, assignments=None, centers=None):
    if assignments is None:
        assignments = [0] * len(X)
    fig = plt.figure(figsize=(14,8))
    cmap = ListedColormap(['red', 'green', 'blue'])
    plt.scatter(X, Y, c=assignments, cmap=cmap)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c=range(len(centers)), marker='+', s=400, cmap=cmap)  
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



file = os.sep.join(['seeds_dataset.txt'])
data = pd.read_csv(file, sep="\t", header=None, names=["Area", "Perimeter", "Compactness", "Length of Kernel","Width of Kernel","Asymmetry coefficient"," Length of kernel groove","h"])
data=data.drop(columns=['h'])

tf.logging.set_verbosity(tf.logging.ERROR)
kmeans = tf.contrib.learn.KMeansClustering(num_clusters=3, relative_tolerance=0.0001)
kfit = kmeans.fit(input_fn=input_fn)


clusters = kmeans.clusters()
assignments = list(kmeans.predict_cluster_idx(input_fn=input_fn))

#ScatterPlot(data.Area, data.Perimeter, assignments, clusters)


#k means error reporting
#find the most popular per 70
#print(type(clusters))

cluster_labels_list = assignments
#print(type(cluster_labels_list))
#print(cluster_labels_list[0])

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


print(cluster_labels_list[:70].count(mode1))
print(70-cluster_labels_list[:70].count(mode1))
print(cluster_labels_list[70:140].count(mode2))
print(70-cluster_labels_list[70:140].count(mode2))
print( cluster_labels_list[140:].count(mode3))
print(70-cluster_labels_list[140:].count(mode3))


print(str(Ttrue) + "/" + str(Ttrue + Tfalse))
print("Accuracy: " + str(Ttrue / (Ttrue + Tfalse)))