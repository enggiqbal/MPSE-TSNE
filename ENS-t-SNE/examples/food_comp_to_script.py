# %%
import numpy as np 
import pandas as pd 
import pylab as plt
import mview

# %%
data = pd.read_csv("food_comp_processed.csv")
data.head()

# %%
tab10 = {0: "red", 
        1: "blue",
        2: "orange",
        3: "tab:red",
        4: "tab:purple"}

# %%
from sklearn.cluster import KMeans


labels = data[['Shrt_Desc']].to_numpy()
shrt_labels = np.array([str(lbl) for lbl in labels])


x1 = data[['Water_(g)','Vit_E_(mg)','Sodium_(mg)','Lipid_Tot_(g)','Energ_Kcal']].to_numpy()
x2 = data[['Protein_(g)', 'Vit_B6_(mg)', 'Vit_B12_(µg)', 'Vit_D_µg']].to_numpy()
X = data.drop(['Unnamed: 0', "Shrt_Desc"],axis=1).to_numpy()

C = KMeans(3).fit_predict(X / np.max(X,axis=0))


# %%
keep_labels = [
    "TURKEY,WHL,MEAT ONLY,RAW",
    "SOUP,CHICK BROTH,CND,COND",
    "TOFU,FIRM,PREP W/CA SULFATE&MAGNESIUM CHLORIDE (NIGARI)",
    "BEVERAGES,H2O,TAP,DRINKING",
    "RESTAURANT,CHINESE,VEG LO MEIN,WO/ MEAT",
    "BEEF,CHUCK FOR STEW,LN & FAT,ALL GRDS,RAW",
    "POTATOES,BKD,FLESH,W/SALT",
    "BEANS,KIDNEY,RED,MATURE SEEDS,CND,SOL & LIQUIDS", 
    "APPLES,RAW,WITH SKIN",
    "BANANAS,RAW"
]
in_eng = [
    'Turkey',
    'Chicken Broth',
    'Tofu',
    'Tap Water',
    'Veg Lo Mein',
    'Beef Chuck',
    'Baked Potato',
    'Kidney Beans',
    'Apple',
    'Banana'
]

eng_map = dict(zip(keep_labels,in_eng))

labels = data[['Shrt_Desc']].to_numpy().reshape(-1)
idx = list()
for i,l in enumerate(labels):
    if l in keep_labels: idx.append(i)

shrt_labels = [eng_map[l] if i in idx else "" for i,l in enumerate(labels)]


# %%
x1 /= np.max(x1)
x2 /= np.max(x2,axis=0)

# %%
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
x1 = pairwise_distances(x1)
# x2 = pairwise_distances(x2)

# avg = (x1+x2)/2
# init = PCA(n_components=3).fit_transform(avg)

# %%
proj1 = np.array([[1,0,0],
                  [0,1,0]])
proj2 = np.array([[0,1,0],
                  [0,0,1]])                  

# %%
# mv = mview.mpse_tsne([x1,x2],perplexity=500,
#                     iters=500, smart_init=True,
#                     show_plots=False,initial_projections=[proj1,proj2],
#                     initial_embedding=init)


# # %%
# fig, (ax1,ax2) = plt.subplots(1,2)
# y1 = mv.images[0,:,:]
# y2 = mv.images[1,:,:]
# m = ['^', 'o']
# for ax,emb in zip([ax1,ax2],[y1,y2]):
#     x , y = emb[:,0], emb[:,1]
#     for px,py,c,txt in zip(x,y, C,shrt_labels):
#         ax.scatter(px,py,c=tab10[c], alpha=1)
#         ax.text(px,py,txt,fontsize=14)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.set_size_inches(8,8)
# # fig.tight_layout()
# plt.show()

# %%
from sklearn.manifold import TSNE
Y = TSNE(perplexity=160,init='pca',learning_rate='auto').fit_transform(X / np.max(X,axis=0))
#Y = PCA(2).fit_transform(X / np.max(X,axis=0))

fig, ax = plt.subplots()
for x,y, c in zip(Y[:,0],Y[:,1],C):
    ax.scatter(x,y,c=tab10[c])
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("figs/better-food-tsne.png")

Y = TSNE(perplexity=50,learning_rate='auto',init='pca').fit_transform(x2)

C1 = [tab10[c] for c in C]
plt.scatter(Y[:,0],Y[:,1],c=C1)
for i in range(Y.shape[0]):
    ax.annotate(shrt_labels[i],(Y[i,0],Y[i,1]) if shrt_labels[i] != "Kidney Beans" else (Y[i,0],Y[i,1]+0.2),
                        textcoords="offset points",
                        xytext=(0,4),ha='center',fontsize='small',
                        horizontalalignment='center',color='black')

plt.show()