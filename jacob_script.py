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


import itertools

for c,i in enumerate(itertools.product([0,1],repeat=4)):
    mask = [j for j in range(4) if i[j]]
    if len(mask) > 2: continue
    print(len(mask))
    
    x1 = np.delete(X,mask,axis=1)

    e1 = TSNE(perplexity=20,init='pca' if len(mask) < 3 else 'random',learning_rate='auto').fit_transform(x1)
    scatter = pylab.scatter(e1[:,0],e1[:,1],20, labels)
    elem,_ = scatter.legend_elements()
    pylab.legend(elem,names)
    pylab.savefig('penguins/iter_num{}.png'.format(c))
    pylab.clf()

# mview.mpse_tsne([x1,x2],
#                 perplexity=30,
#                 verbose=2,sample_colors=[labels1,labels], sample_classes=[names1,names],show_plots=True,iters=200)





