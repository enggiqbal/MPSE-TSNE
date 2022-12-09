import os, sys
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
from collections import Counter

path = os.path.dirname(os.path.realpath(__file__))
data = os.path.join(path,'data')

#book chapters are organized by volume, under 'data/' folder
folders = ['volume1','volume2','volume3']

#count occurrences of characters in book:
occurrances = Counter()
for folder in folders:
    directory = os.path.join(data, folder)
    for filename in os.listdir(directory):
        f = open(os.path.join(directory, filename),'r')
        content = f.readlines()
        for line in content:
            name1, name2 = line.strip().split('\t')
            occurrances[name1] += 1
            occurrances[name2] += 1
        f.close()

#list of characters (in order of occurrances):
characters = [name for name,count in occurrances.most_common()]
appearences = [count for name,count in occurrances.most_common()]
n_characters = len(characters) #118
n_apperences = sum(appearences) #8066
#characters w/ at least 10 appearences: 0-40
#characters w/ at least 6 appearences: 0-48

counts = np.zeros((3,n_characters,n_characters),dtype=int)
for i in range(3):
    directory = os.path.join(data, folders[i])
    for filename in os.listdir(directory):
        f = open(os.path.join(directory,filename),'r')
        content = f.readlines()
        for line in content:
            name1, name2 = line.strip().split('\t')
            j= characters.index(name1)
            k = characters.index(name2)
            counts[i,j,k] += 1
            counts[i,k,j] += 1
        f.close()

edges = []
for i in range(3):
    edges_i = []
    for j in range(118):
        for k in range(j+1,118):
            if counts[i,j,k] != 0:
                edges_i.append([j,k])
    edges.append(edges_i)

distances = np.empty((3,118,118))
for i in range(3):
    distances[i] = 1.0 / np.maximum(0.2,counts[i])
    #distances[i] /= np.maximum(1,distances[i].sum(axis=1))[:,None]
    import scipy.sparse.csgraph as csgraph
    distances[i] = csgraph.shortest_path(distances[i])
    distances[i] += np.random.normal(0,0.01,(118,118))

names = list(range(10))+[None]*108
