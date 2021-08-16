import os, sys
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# sys.path.insert(1,'../..')
sys.path.insert(1,'/Users/vahanhuroyan/Documents/MPSE/MPSE-TSNE')

import mview


filePath = 'mview/temp/embedding.csv'

df1 = pd.read_csv(filePath)
a = df1.values

print(len(a))
print(a[0])

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# ax.scatter3D(a[0], a[1], a[3], cmap='Greens');

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # Data for a three-dimensional line
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
# ax.plot3D(xline, yline, zline, 'gray')
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

# plt.show()