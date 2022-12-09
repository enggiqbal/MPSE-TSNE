import os, sys
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt

# sys.path.insert(1,'../..')
sys.path.insert(1,'/Users/vahanhuroyan/Documents/MPSE/MPSE-TSNE')

import mview


#file and pandas object with data:
path = os.path.dirname(os.path.realpath(__file__))
file = path+'/colleges.csv'

print(file)

# with open(file) as csvfile:
#     df = pd.read_csv(csvfile)
# print(df)

# df1 = pd.read_csv(file)

# df1 = pd.read_csv(file, usecols = ['mpg',  'cylinders', 'displacement'])
# df2 = pd.read_csv(file, usecols = ['horsepower',  'weight', 'acceleration'])

df1 = pd.read_csv(file, usecols=[0,1,2])
df2 = pd.read_csv(file, usecols=[3,4,5])
# df3 = pd.read_csv(file, usecols=[6,7])

# df2["horsepower"] = [float(str(i).replace(",", "")) for i in df2["horsepower"]]

# df1.to_csv(path + '/car_mpg_1.csv')

a = df1.values
b = df2.values
# c = df3.values



# Normalizing the columns by the max element
maxVecA = np.amax(a, axis = 0)
maxVecB = np.max(b, axis = 0)
# maxVecC = np.max(c, axis = 0)


a = np.divide(a.T, maxVecA.reshape(len(maxVecA), 1)).T
b = np.divide(b.T, maxVecB.reshape(len(maxVecB), 1)).T
# c = np.divide(c.T, maxVecC.reshape(len(maxVecC), 1)).T



data = [a, b]

# data = [a, b, c]

results = mview.mpse_tsne(data, perplexity=30, show_plots=True, save_results=True, verbose=2)