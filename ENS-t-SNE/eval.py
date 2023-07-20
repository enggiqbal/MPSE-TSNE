import numpy as np 
import pandas as pd 
import pylab as plt
import mview

from umap import UMAP
from sklearn.manifold import TSNE

import matplotlib.patheffects as PathEffects


tab10 = {0: "red", 
        1: "blue",
        2: "orange",
        3: "tab:red",
        4: "tab:purple"}

def plot_3d(Emb: np.array, C:np.array,title=None):
    #Plot 3D
    fig= plt.figure(figsize=(5,4))
    ax = fig.add_subplot(1,1,1,projection='3d')

    x,y,z = Emb[:,0], Emb[:,1], Emb[:,2]
    ax.scatter3D(x,y,z,c=C,alpha=0.9)

    ax.grid(color='r')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.setp(ax.spines.values(), color='blue')
    if title: 
        plt.savefig(title)
    else: plt.show()


def plot2d(data,labels,title=None):
    fig, axes = plt.subplots(1,len(data))
    for i,(ax, x) in enumerate(zip(axes,data)):
        ax.scatter(x[:,0],x[:,1], c=labels[i])
    if title: plt.savefig(title)
    else: plt.show()

from metrics import compute_all_metrics, compute_all_3dmetrics
from new_enstsne import get_clusters, ENSTSNE
from sklearn.metrics import pairwise_distances

def eval(dists, labels,full):
    #Compute enstsne, old, new
    old_ens = mview.mpse_tsne(dists)
    old_ens3d = old_ens.embedding
    old_ens = old_ens.images
    
    new_ens = ENSTSNE(dists, 30, labels, early_exaggeration=5)
    new_ens.gd(1000,lr=50)
    enstsne3d = new_ens.get_embedding()
    new_ens = new_ens.get_images()

    #Compute mpse 
    mv = mview.basic(dists, verbose=2, smart_initialize=True, max_iter=1000,learning_rate=0.001)
    mpse3d = mv.embedding
    mpse = mv.images
    
    #Compute tsne, umap 3d 
    from sklearn.manifold import TSNE 
    tsne = [TSNE(metric="precomputed").fit_transform(d) for d in dists]
    tsne3d = TSNE(n_components=3,metric="precomputed").fit_transform(full)
    
    from umap import UMAP
    umap = [UMAP(metric="precomputed").fit_transform(d) for d in dists]
    umap3d = UMAP(n_components=3, metric="precomputed").fit_transform(full)

    from sklearn.manifold import MDS 
    mds = [MDS(metric="precomputed").fit_transform(d) for d in dists]
    mds3d = MDS(n_components=3, metric="precomputed").fit_transform(full)


    #Compute metrics
    algs = ["old_ens", "ens-t-sne", "mpse", "tsne", "umap", "mds"]
    embeddings = [old_ens, new_ens, mpse, tsne, umap, mds]
    vals = {f"view-{i}": {
        alg: compute_all_metrics(embedding[i],dists[i],labels[i]) for alg, embedding in zip(algs,embeddings)
    } for i,_ in enumerate(dists) }

    embeddings3d = [old_ens3d, enstsne3d, mpse3d, tsne3d, umap3d, mds3d]
    vals["3d"] = {alg: compute_all_3dmetrics(embedding,full, labels) for alg, embedding in zip(algs,embeddings3d)}
    
    for key in vals: 
        print(f"{key} -> {vals[key].keys()}")

    return vals


from new_enstsne import load_clusters, load_penguins, load_auto, load_food
if __name__ == "__main__":

    funcs = [
        ["synthetic", lambda : load_clusters(300, [10,10,10], [2,3,4])],
        # ["penguins", lambda : load_penguins()],
        # ["auto", lambda: load_auto()],
        # ["food", lambda: load_food()]
        ]
    
    for name, f in funcs:
        dists, labels, X = f()

        results = eval(dists,labels,pairwise_distances(X))

        import pickle 
        # open a file, where you ant to store the data
        file = open(f'results/{name}.pkl', 'wb')

        # dump information to that file
        pickle.dump(results, file)

        # close the file
        file.close()    



    """
    Why are we losing in cluster distance? All clusters equidistant.
    """