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
file = path+'/auto-mpg_mod.csv'

print(file)

# with open(file) as csvfile:
#     df = pd.read_csv(csvfile)
# print(df)

# df1 = pd.read_csv(file)

# df1 = pd.read_csv(file, usecols = ['mpg',  'cylinders', 'displacement'])
# df2 = pd.read_csv(file, usecols = ['horsepower',  'weight', 'acceleration'])

df1 = pd.read_csv(file, usecols=[0,1,2])

df2 = pd.read_csv(file, usecols=[3,4,5])

# df2["horsepower"] = [float(str(i).replace(",", "")) for i in df2["horsepower"]]

# df1.to_csv(path + '/car_mpg_1.csv')

a = df1.values

b = df2.values

# a.astype(float)
# b.astype(float)


print(len(a), len(b))
print(len(a[0]), len(b[0]))
print(a[0])
print(b[0])



# print(df1)

# print([df1, df2])

# pd.to_csv(df1, path + '1.csv')

# df1.to_csv(path + '/car_mpg_1.csv',  header = None, index=False)

print(type(a))
print(type(b))

data = [a, b]
results = mview.mpse_tsne(data, perplexity=30, show_plots=True, verbose=2)
