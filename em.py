#EM clustering algorithm using Gaussian Mixture implementation

import os
import numpy as np
import pandas as pd
from statistics import mode

import sklearn
from sklearn.mixture import gaussian_mixture

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def main():
    dataFrame = import_data()
    dataList, dataClasses = cleanData(dataFrame)
    predLabels = kmeans(dataList)
    validation(predLabels)



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



def kmeans(data):
    # Initialize the clusterer to make 3 clusters
    # seed of 10 for reproducibility.
    clusterer = gaussian_mixture.GaussianMixture(n_components=3)
    clusterer.fit(data)
    cluster_labels = clusterer.predict(data)
    cluster_labels_list = cluster_labels.tolist()
    #print(type(cluster_labels_list))
    return cluster_labels_list



def validation(labels):
    #k means error reporting
    #find the most popular per 70
    #print(type(cluster_labels))
    mode1 = mode(labels[:70])
    mode2 = mode(labels[70:140])
    mode3 = mode(labels[140:])
    #print(mode1,mode2,mode3)

    Ttrue = 0
    Tfalse = 0
    Ttrue += labels[:70].count(mode1)
    Tfalse += 70-labels[:70].count(mode1)
    Ttrue += labels[70:140].count(mode2)
    Tfalse += 70-labels[70:140].count(mode2)
    Ttrue += labels[140:].count(mode3)
    Tfalse += 70-labels[140:].count(mode3)

    print(str(Ttrue) + "/" + str(Ttrue + Tfalse))
    print("Accuracy: " + str(Ttrue / (Ttrue + Tfalse)))
    #print(labels)

    """
    print(labels[:70].count(mode1))
    print(70-labels[:70].count(mode1))
    print(labels[70:140].count(mode2))
    print(70-labels[70:140].count(mode2))
    print( labels[140:].count(mode3))
    print(70-labels[140:].count(mode3))
    """


main()