# %%
import numpy as np 
import pandas as pd 
import pylab as plt
import mview

# %%
data = pd.read_csv("food_comp_processed.csv")
data.head()

# %%
tab10 = {0: "tab:blue", 
        1: "tab:orange",
        2: "tab:green",
        3: "tab:red",
        4: "tab:purple"}

# %%
from sklearn.cluster import KMeans


labels = data[['Shrt_Desc']].to_numpy()
shrt_labels = np.array([str(lbl)[2:10] for lbl in labels])


x1 = data[['Water_(g)','Vit_E_(mg)','Sodium_(mg)','Lipid_Tot_(g)','Energ_Kcal']].to_numpy()
x2 = data[['Protein_(g)', 'Vit_B6_(mg)', 'Vit_B12_(µg)', 'Vit_D_µg']].to_numpy()
X = data.drop(['Unnamed: 0', "Shrt_Desc"],axis=1).to_numpy()

C = KMeans(3).fit_predict(X / np.max(X,axis=0))


# %%
x1 /= np.max(x1)
x2 /= np.max(x2)

# %%
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
x1 = pairwise_distances(x1)
x2 = pairwise_distances(x2)

avg = (x1+x2)/2
init = PCA(n_components=3).fit_transform(avg)

# %%
proj1 = np.array([[1,0,0],
                  [0,1,0]])
proj2 = np.array([[0,1,0],
                  [0,0,1]])                  

# %%
mv = mview.mpse_tsne([x1,x2],perplexity=400,
                    iters=500, smart_init=True,
                    show_plots=False,initial_projections=[proj1,proj2],
                    initial_embedding=init)


# %%
fig, (ax1,ax2) = plt.subplots(1,2)
y1 = mv.images[0,:,:]
y2 = mv.images[1,:,:]
m = ['^', 'o']
for ax,emb in zip([ax1,ax2],[y1,y2]):
    x , y = emb[:,0], emb[:,1]
    for px,py,c in zip(x,y, C):
        ax.scatter(px,py,c=tab10[c], alpha=1)
    ax.set_xticks([])
    ax.set_yticks([])

fig.set_size_inches(8,8)
fig.tight_layout()

# %%
from sklearn.manifold import TSNE
Y = TSNE(perplexity=80,init='pca',learning_rate='auto').fit_transform(X / np.max(X,axis=0))

for x,y, c in zip(Y[:,0],Y[:,1],C):
    plt.scatter(x,y,c=tab10[c])
plt.show()


