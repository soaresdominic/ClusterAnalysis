#kmeans clustering algorithm implementation

import os
import numpy as np
import pandas as pd
from statistics import mode
from statistics import mean
import random
import copy

import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def main():
    dataFrame = import_data()
    dataList, dataClasses = cleanData(dataFrame)
    accuracies = []
    for i in range(10):
        predLabels = kmeans(dataList)
        accuracies.append(validation(predLabels, len(predLabels)))
    print(accuracies)
    print("The average accuracy is: " + str(mean(accuracies)))


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

    n1 = int(size/3)
    n2 = n1*2

    mode1 = mode(labels[:n1])
    mode2 = mode(labels[n1:n2])
    mode3 = mode(labels[n2:])
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