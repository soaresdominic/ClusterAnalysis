import os, numpy as np, pandas as pd
file = os.sep.join(['seeds_dataset.txt'])
data = pd.read_csv(file, sep="\t", header=None, names=["Area", "Perimeter", "Compactness", "Length of Kernel","Width of Kernel","Asymmetry coefficient"," Length of kernel groove","Label"])

#data=data.drop(columns=['h'])
data.head()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import somoclu
#%matplotlib inline 
###################
c1 = list(data['Area'].values)
c2 = list(data['Perimeter'].values)
c3 = list(data['Compactness'].values)  #more features can be added like this
c4 = list(data['Length of Kernel'].values)
c5 = list(data[' Length of kernel groove'].values)
data_somo=[]
for i in range(0, len(c1)):
    data_somo.append([c1[i],c2[i]])  #append all the features here
data_somo=np.array(data_somo)

colors = ["red"] * 70
colors.extend(["green"] * 70)
colors.extend(["blue"] * 70)
fig = plt.figure()
ax = Axes3D(fig)
labels =np.array(data['Label'].values)
#ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)
ax.scatter(c1,c4,c5, c=labels)
plt.show()
#labels = range(150)
print(labels)

