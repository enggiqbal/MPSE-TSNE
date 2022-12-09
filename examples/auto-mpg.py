import os, sys
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt

# sys.path.insert(1,'../..')
#sys.path.insert(1,'/Users/vahanhuroyan/Documents/MPSE/MPSE-TSNE')

import mview


#file and pandas object with data:
#path = os.path.dirname(os.path.realpath(__file__))
file = "mview/samples/car_mpg/auto-mpg_mod.csv"

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
df = pd.read_csv(file, usecols=[0,1,2,3,4,5])

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

# results = mview.mpse_tsne(data, perplexity=30, show_plots=True, save_results=True, verbose=2)
print(df.to_dict().keys())
data = df.to_dict()
clr_map = dict()
for i,item in enumerate(data['cylinders'].values()):
    if item <= 4: clr_map[i] = "red"
    elif item <= 6: clr_map[i] = "blue"
    else: clr_map[i] = "orange"

weight = df['weight'].to_numpy()
quarter, half, seventy = np.quantile(weight,[0.25,0.5,0.75])

shape_map = dict()
for i,item in enumerate(data["weight"].values()):
    if item <= quarter: shape_map[i] = "D"
    elif item <= half: shape_map[i] =  "x"
    elif item <= seventy: shape_map[i] = "o"
    else: shape_map[i] = "^"

# fig, (ax1,ax2) = plt.subplots(1,2)

# Y = results.images
# y1 = Y[0,:,:]
# y2 = Y[1,:,:]

# for ax,emb in zip([ax1,ax2],[y1,y2]):
#     x , y = emb[:,0], emb[:,1]
#     for i, (px,py) in enumerate(zip(x,y)):
#         ax.scatter(px,py,marker=shape_map[i],c=clr_map[i], alpha=0.5)
#     ax.set_xticks([])
#     ax.set_yticks([])

# plt.show()


# fig= plt.figure(figsize=(5,4))
# ax = fig.add_subplot(1,1,1,projection='3d')

# X = np.loadtxt("mview/temp/embedding.csv")

# perspectives = list()
# for k in range(mv.n_perspectives):
#     Q = mv.projections[k]
#     q = np.cross(Q[0],Q[1])
#     perspectives.append(q)

# q = perspectives
# for k in range(len(q)):
#     ind = np.argmax(np.sum(q[k]*X,axis=1))
#     m = np.linalg.norm(X[ind])/np.linalg.norm(q[k])
#     ax.plot([0,m*q[k][0]],[0,m*q[k][1]],[0,m*q[k][2]],'--',
#             linewidth=4.5,
#             color='gray')

# N = len(X)

# x,y,z = X[:,0], X[:,1], X[:,2]
# for i, (px,py,pz) in enumerate(zip(x,y,z)):
#     ax.scatter3D(px,py,pz,c=clr_map[i],marker=shape_map[i], alpha=0.5)

# ax.grid(color='r')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False
# plt.setp(ax.spines.values(), color='blue')
# plt.show()

X = df.to_numpy()
from sklearn.manifold import TSNE 
Y = TSNE(perplexity=30, learning_rate='auto',init='pca').fit_transform(X / np.max(X,axis=0))

fig,ax = plt.subplots()
x , y = Y[:,0], Y[:,1]
for i, (px,py) in enumerate(zip(x,y)):
    ax.scatter(px,py,marker=shape_map[i],c=clr_map[i], alpha=0.5)
ax.set_xticks([])
ax.set_yticks([])
plt.show()