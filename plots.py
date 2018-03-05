import os
import pandas as pd
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


file = os.sep.join(['seeds_dataset.txt'])
data = pd.read_csv(file, sep="\t", header=None, names=["Area", "Perimeter", "Compactness", "Length of Kernel","Width of Kernel","Asymmetry coefficient","Length of kernel groove","h"])
data=data.drop(columns=['h'])

fig = pyplot.figure()
ax = Axes3D(fig)

d1 = list(data.get("Area"))
d2 = list(data.get("Perimeter"))
d3 = list(data.get("Compactness"))
d4 = list(data.get("Length of Kernel"))
d5 = list(data.get("Width of Kernel"))
d6 = list(data.get("Asymmetry coefficient"))
d7 = list(data.get("Length of kernel groove"))

sequence_containing_x_vals = d1
sequence_containing_y_vals = d2
sequence_containing_z_vals = d3

ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
pyplot.show()