#File for general DM stuff

import os

import numpy as np
import pandas as pd
import copy
from statistics import mode

import sklearn
from sklearn.cluster import KMeans
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

    tempL.append(instance[4])
    tempL.append(instance[6])
    
    X.append(tempL)
#print(finalA)
#print(finalB)

X = np.array(X)
#print(X)

#scikitlearn.org code
# Create a subplot with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
# The 1st subplot is the silhouette plot
# The silhouette coefficient can range from -1, 1 but in this example all
# lie within [-0.1, 1]
ax1.set_xlim([-0.1, 1])
n_clusters = 3
# The (n_clusters+1)*10 is for inserting blank space between silhouette
# plots of individual clusters, to demarcate them clearly.
ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

# Initialize the clusterer with 3 value and a random generator
# seed of 10 for reproducibility.
clusterer = KMeans(n_clusters=n_clusters, n_init=50, max_iter=1000, random_state=1)
cluster_labels = clusterer.fit_predict(X)
print(cluster_labels)



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




# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(X, cluster_labels)
print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X, cluster_labels)

y_lower = 10
for i in range(n_clusters):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = \
        sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.spectral(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples


ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# 2nd Plot showing the actual clusters formed
colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
#print(type(X))
#print(colors)
ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

# Labeling the clusters
centers = clusterer.cluster_centers_
# Draw white circles at cluster centers
ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
            c="white", alpha=1, s=200, edgecolor='k')

for i, c in enumerate(centers):
    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                s=70, edgecolor='k')

ax2.set_title("The visualization of the clustered data.")
ax2.set_xlabel("Feature space for the 1st feature")
ax2.set_ylabel("Feature space for the 2nd feature")

plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                "with n_clusters = %d" % n_clusters),
                fontsize=14, fontweight='bold')

plt.show()









"""
for i in range(0,6):
    finalA = []
    finalB = []
    for instance in dataVals:
        finalA.append(instance[i])
        finalB.append(instance[i+1])
    #print(finalA)

    plt.figure(figsize=(12,12))
    ax = plt.axes()
    ax.scatter(finalA, finalB)
    plt.show()
"""
