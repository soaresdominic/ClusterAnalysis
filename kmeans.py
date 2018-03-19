#Cluster Analysis
#kmeans clustering algorithm implementation
#USAGE:
#     python3 kmeans.py
#     python3 kmeans.py validation
#validation - shows the silhouette plots and values for the two attributes selected on line 36

import os
import numpy as np
import pandas as pd
from statistics import mode
from statistics import mean
import random
import sys
import scipy.stats as scp

import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

def main():
    dataFrame = import_data()
    dataList, dataClasses = cleanData(dataFrame)
    accuracies = []
    for i in range(40):
        predLabels = kmeans(dataList)
        accuracies.append(validation(predLabels, len(predLabels)))
    #print(accuracies)
    print("Accuracy: " + str(100*(round(mean(accuracies),4))) + "%")

    if(len(sys.argv) > 1):
        if(sys.argv[1] == "validation"):
            showSilhouette(dataList, dataClasses, [2,4])
        else:
            print("Argument Error.")


def import_data():
    file = os.sep.join(['seeds_dataset.txt'])
    data = pd.read_csv(file, sep="\t", header=None, names=["Area", "Perimeter", "Compactness", "Length of Kernel","Width of Kernel","Asymmetry coefficient"," Length of kernel groove","h"])
    return data


def cleanData(dataFrame):
    data_classes = []
    for instance in dataFrame.values:
        data_classes.append(instance[-1])
    dataFrame=dataFrame.drop(columns=['h'])
    dataVals = dataFrame.values

    data = []
    for instance in dataVals:
        data.append(instance.tolist())
    data = np.array(data)
    #print(data)
    return data, data_classes


def showSilhouette(data, cluster_labels, attributes):
    X = []
    for instance in data:
        X.append([instance[attributes[0]], instance[attributes[1]]])
    X = np.array(X)
    #print(X)
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

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters

    clusterer_s = KMeans(n_clusters=3, random_state=10)
    cluster_labels_s = clusterer_s.fit_predict(X)

    #print(cluster_labels_s)

    silhouette_avg = silhouette_score(X, cluster_labels_s)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels_s)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels_s == i]

        ith_cluster_silhouette_values.sort()

        #print(ith_cluster_silhouette_values)

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.Spectral(float(i) / n_clusters)
        #print(y_lower)
        #print(y_upper)
        #print(np.arange(y_lower, y_upper))
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
    colors = cm.Spectral(cluster_labels_s.astype(float) / n_clusters)
    #print(type(X))
    #print(colors)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer_s.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=70, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on seed data "
                    "with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')
    plt.show()


def kmeans(data):
    # Initialize the clusterer to make 3 clusters
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=3, random_state=10)
    train, test = seperateData(data)
    clusterer.fit(train)  #train
    cluster_labels = clusterer.predict(test)  #test
    cluster_labels_list = cluster_labels.tolist()
    #print(cluster_labels_list)
    return cluster_labels_list


#data is a list of lists
def seperateData(data):
    data = data.tolist()
    dataSeperate = [data[:70],data[70:140], data[140:]]
    test = []
    for d in dataSeperate:  #3 sets of 70 instances
        for i in range(10):  #grab 10 from each set
            r = random.randint(0,len(d)-1)
            test.append(d.pop(r))
    #print(data)
    #print(test)
    return data, test


def validation(labels, size):
    #k means error reporting
    #find the most popular per 70
    #print(type(cluster_labels))
    #print(labels)

    n1 = int(size/3)
    n2 = n1*2

    mode1 = scp.mode(labels[:n1])[0]
    mode2 = scp.mode(labels[n1:n2])[0]
    mode3 = scp.mode(labels[n2:])[0]
        
    #print(mode1,mode2,mode3)

    Ttrue = 0
    Tfalse = 0
    Ttrue += labels[:n1].count(mode1)
    Tfalse += n1-labels[:n1].count(mode1)
    Ttrue += labels[n1:n2].count(mode2)
    Tfalse += n1-labels[n1:n2].count(mode2)
    Ttrue += labels[n2:].count(mode3)
    Tfalse += n1-labels[n2:].count(mode3)

    #print(str(Ttrue) + "/" + str(Ttrue + Tfalse))
    #print("Accuracy: " + str(Ttrue / (Ttrue + Tfalse)))
    #print(labels)
    return (Ttrue / (Ttrue + Tfalse))


main()