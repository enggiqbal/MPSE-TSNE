import mview
import graph_tool.all as gt
import numpy as np

#Area,Perimeter,MajorAxisLength,MinorAxisLength,AspectRation,Eccentricity,ConvexArea,EquivDiameter,Extent,Solidity,roundness,Compactness,ShapeFactor1,ShapeFactor2,ShapeFactor3,ShapeFactor4,Class


from sklearn.manifold import TSNE
import pylab

import random

data = np.loadtxt('palmerpenguins.csv',delimiter=',',dtype='U100',skiprows=1)



names,labels = np.unique(data[:,2],return_inverse=True)
names1,labels1 = np.unique(data[:,1],return_inverse=True)
sex,sex_val = np.unique(data[:,7],return_inverse=True)

X = np.delete(data, [0,1,2,7,8], axis=1).astype(float)
X /= X.max(axis=0)


e1 = TSNE(perplexity=20,init='pca',learning_rate='auto').fit_transform(X)
scatter = pylab.scatter(e1[:,0],e1[:,1],20, labels)
elem,_ = scatter.legend_elements()
pylab.legend(elem,names1)
pylab.show()
# for p in [2,10,20,30,40,50,60,80,100,120,140,180,200]:
# for _ in range(20):
dims = [0,1,2,3]
dim1 = [1,2]
dim2 = [0,3]
#dim2 = [i for i in dims if i not in dim1]

x1 = np.delete(X, dim1 ,axis=1)
x2 = np.delete(X, dim2, axis=1)
# x2 = sex_val
# x2 = np.reshape(x2, (-1,1))

mview.mpse_tsne([x1,x2],
                perplexity=30,
                verbose=2,sample_colors=[labels1,labels], sample_classes=[names1,names],show_plots=True,iters=500)


