import os, sys
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt

#file and pandas object with data:
path = os.path.dirname(os.path.realpath(__file__))
file = path+'/auto-mpg.csv'

print(file)

with open(file) as csvfile:
    df = pd.read_csv(csvfile)

print(df)