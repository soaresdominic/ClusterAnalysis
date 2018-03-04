#Tensorflow deep clustering algorithm implementation

import os
import pandas as pd
from statistics import mode
import os
import pandas as pd
import tensorflow as tf

data = None
def main():
    global data
    data = import_data()
    predLabels = Tcluster()
    validation(predLabels)



def import_data():
    file = os.sep.join(['seeds_dataset.txt'])
    data = pd.read_csv(file, sep="\t", header=None, names=["Area", "Perimeter", "Compactness", "Length of Kernel","Width of Kernel","Asymmetry coefficient"," Length of kernel groove","h"])
    return data


def input_fn():
    global data
    return tf.constant(data.as_matrix(), tf.float32, data.shape), None



def Tcluster():
    tf.logging.set_verbosity(tf.logging.ERROR)
    kmeans = tf.contrib.learn.KMeansClustering(num_clusters=3, relative_tolerance=0.0001)
    kfit = kmeans.fit(input_fn=input_fn)
    clusters = kmeans.clusters()
    assignments = list(kmeans.predict_cluster_idx(input_fn=input_fn))
    return assignments



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