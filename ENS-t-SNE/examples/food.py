import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv('food_comp_processed.csv')
X = data.drop(['Shrt_Desc','Unnamed: 0'], axis=1).to_numpy()
print(X)

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import random
import mview 

# np.random.seed(12345)

C = data[['class']].to_numpy()

labels = data[['Shrt_Desc']].to_numpy()
shrt_labels = np.array([str(lbl)[2:10] for lbl in labels])


x1 = data[['Water_(g)','Vit_E_(mg)','Sodium_(mg)','Lipid_Tot_(g)','Energ_Kcal']].to_numpy()
x2 = data[['Protein_(g)', 'Vit_B6_(mg)', 'Vit_B12_(µg)', 'Vit_D_µg']].to_numpy()
#x2 = data[['Carbohydrt_(g)','Fiber_TD_(g)', 'Calcium_(mg)', 'Iron_(mg)','Magnesium_(mg)','Manganese_(mg)']].to_numpy()

# ind = np.random.choice(x1.shape[0], 200, replace=False)
# x1 = x1[ind]
# x2 = x2[ind]
# shrt_labels = shrt_labels[ind]


X = np.concatenate((x1,x2),axis=1)

C = KMeans(n_clusters=3).fit_predict(X / np.max(X))

# Y1 = TSNE().fit_transform(x1)
# Y2 = TSNE(perplexity=40).fit_transform(x2)
# Y3 = TSNE(perplexity=30).fit_transform(X)
# import pylab 
# # pylab.scatter(Y1[:,0],Y1[:,1],20,C)
# # pylab.savefig('figs/food_tsnex1.png')
# # pylab.clf()
# # pylab.scatter(Y2[:,0],Y2[:,1],20,C)
# # pylab.savefig('figs/food_tsnex2.png')
# # pylab.clf()
# pylab.scatter(Y3[:,0],Y3[:,1],20,C)
# pylab.savefig("figs/food_union.png")

x1 /= np.max(x1)
x2 /= np.max(x2)

from sklearn.metrics import pairwise_distances
x1 = pairwise_distances(x1)
x2 = pairwise_distances(x2)

avg = (x1 + x2) / 2

from sklearn.decomposition import PCA

smart = PCA(n_components=3).fit_transform(avg)





proj1 = np.array([[1,0,0],
                 [0,1,0]])

proj2 = np.array([[0,1,0],
                  [0,0,1]])


#
print(shrt_labels)
# for p in range(100,205,5):
mv = mview.mpse_tsne([x1,x2],perplexity=400,
                    iters=500, sample_colors=[C,C], sample_classes=[C,C],
                    show_plots=False,sample_labes=shrt_labels,verbose=2,save_results=True,initial_embedding=smart,initial_projections=[proj1,proj2])
                    #fixed_projections=[proj,proj])

mv.plot_computations(plot=False)
# mv.plot_embedding(title='final embeding',plot=False)
# mv.plot_images(plot=False)#edges=edges, labels=labels)

# plt.savefig("figs/perplexity-testsp{}.png".format(p))
# plt.clf()

plt.show()

# from mview.mtsne import compare_perplexity

# compare_perplexity([x1,x2])