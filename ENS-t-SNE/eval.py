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

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
markers = ["o", "s", "v", "p", "*", "^" ]
fillstyles = ["full", "none", "top", "bottom", "left", "right"]

markerstyles = [
    colors,
    markers,
    fillstyles
]

import matplotlib

def vis_3d(X, labels, title=None):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')

    pair_labels = np.zeros((X.shape[0], 3),dtype=np.int32)
    for i in range(X.shape[0]):
        for j in range(len(labels)): pair_labels[i,j] = labels[j][i]

    num_clusts = np.max(pair_labels,axis=0)

    for i in range(num_clusts[0] + 1):
        for j in range(num_clusts[1] + 1):
            for k in range(num_clusts[2] + 1):
                emb = X[(pair_labels[:,0] == i) & (pair_labels[:,1] == j) & (pair_labels[:,2] == k)]
                x,y,z = emb[:,0], emb[:,1], emb[:,2]
                style = matplotlib.markers.MarkerStyle(markers[j],fillstyle=fillstyles[k])
                ax.scatter(x,y,z,c=colors[i], marker=style, alpha=0.9)
    plt.xticks(color="w")
    plt.yticks(color="w")
    ax.set_zticks(ax.get_zticks(), [" " for _ in ax.get_zticks()])

    if title: plt.savefig(title)
    else: plt.show()


def vis_2d(data,labels,title=None):
    fig, axes = plt.subplots(1,len(data))

    pair_labels = np.zeros((data[0].shape[0], 3),dtype=np.int32)
    for i in range(data[0].shape[0]):
        for j in range(len(labels)): pair_labels[i,j] = labels[j][i]

    num_clusts = np.max(pair_labels,axis=0)

    l = 0
    for ax, X in zip(axes,data):
        # plt.clf()
        # fig, ax = plt.subplots()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for i in range(num_clusts[0] + 1):
            for j in range(num_clusts[1] + 1):
                for k in range(num_clusts[2] + 1):
                    emb = X[ (pair_labels[:,0] == i) & (pair_labels[:,1] == j) & (pair_labels[:,2] == k) ]
                    x,y = emb[:,0], emb[:,1]
                    style = matplotlib.markers.MarkerStyle(markers[j],fillstyle=fillstyles[k])
                    ax.scatter(x,y,c=colors[i],marker=style,alpha=0.8,linewidths=1)
                    plt.xticks(color="w")
                    plt.yticks(color="w")
        # if title: plt.savefig(f"figs/test{l}.pdf")
        l += 1

    if title: plt.savefig(title)
    else: plt.show()

def vis_2d_per_proj(data,labels,title=None):
    pair_labels = np.zeros((data[0].shape[0], 3),dtype=np.int32)
    for i in range(data[0].shape[0]):
        for j in range(len(labels)): pair_labels[i,j] = labels[j][i]

    num_clusts = np.max(pair_labels,axis=0)

    l = 0
    for X in data:
        plt.clf()
        fig, ax = plt.subplots()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for i in range(num_clusts[0] + 1):
            for j in range(num_clusts[1] + 1):
                for k in range(num_clusts[2] + 1):
                    emb = X[ (pair_labels[:,0] == i) & (pair_labels[:,1] == j) & (pair_labels[:,2] == k) ]
                    x,y = emb[:,0], emb[:,1]
                    style = matplotlib.markers.MarkerStyle(markers[j],fillstyle=fillstyles[k])
                    ax.scatter(x,y,c=colors[i],marker=style,alpha=0.8,linewidths=1)
                    plt.xticks(color="w")
                    plt.yticks(color="w")
        if title: plt.savefig(f"figs/test{l}.pdf")
        else: plt.show()
        l += 1

    if title: plt.savefig(title)
    else: plt.show()    

from metrics import compute_all_metrics, compute_all_3dmetrics
from new_enstsne import get_clusters, ENSTSNE
from sklearn.metrics import pairwise_distances

def eval(dists, labels,full,high_d,dataname=""):
    #Compute enstsne, old, new
    # old_ens = mview.mpse_tsne(dists)
    # old_ens3d = old_ens.embedding
    # old_ens = old_ens.images
    
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
    algs = [ "ens-t-sne", "mpse", "tsne", "umap", "mds"]
    embeddings = [ new_ens, mpse, tsne, umap, mds]
    vals = {f"view-{i}": {
        alg: updated_metrics_2d(embedding[i],dists[i],labels[i]) for alg, embedding in zip(algs,embeddings)
    } for i,_ in enumerate(dists) }

    embeddings3d = [ enstsne3d, mpse3d, tsne3d, umap3d, mds3d]
    vals["3d"] = {alg: updated_metrics_3d(embedding,full, labels) for alg, embedding in zip(algs,embeddings3d)}

    for alg, emb in zip(algs, embeddings3d):
        vis_3d(emb, labels, f"figs/{dataname}_{alg}_3d.pdf")
        np.savetxt(f"embeddings/{dataname}_{alg}_3d.txt",emb)

    for alg, emb in zip(algs,embeddings):
        vis_2d(emb,labels,f"figs/{dataname}_{alg}_2d.pdf")
    
    for key in vals: 
        print(f"{key} -> {vals[key].keys()}")

    return vals

from metrics import updated_metrics_3d, updated_metrics_2d
def eval_small(dists, labels,full,high_d,dataname=""):
    #Compute enstsne, old, new
    # old_ens = mview.mpse_tsne(dists)
    # old_ens3d = old_ens.embedding
    # old_ens = old_ens.images
    
    new_ens = ENSTSNE(dists, 30, labels, early_exaggeration=5)
    new_ens.gd(1000,lr=50)
    enstsne3d = new_ens.get_embedding()
    new_ens = new_ens.get_images()

    #Compute mpse 
    mv = mview.basic(dists, verbose=2, smart_initialize=True, max_iter=1000,learning_rate=0.001)
    mpse3d = mv.embedding
    mpse = mv.images

    #Compute metrics
    algs = [ "ens-t-sne", "mpse"]
    embeddings = [ new_ens, mpse]
    vals = {f"view-{i}": {
        alg: updated_metrics_2d(embedding[i],dists[i],labels[i]) for alg, embedding in zip(algs,embeddings)
    } for i,_ in enumerate(dists) }

    embeddings3d = [ enstsne3d, mpse3d]
    vals["3d"] = {alg: updated_metrics_3d(embedding,full, labels) for alg, embedding in zip(algs,embeddings3d)}

    for alg, emb in zip(algs, embeddings3d):
        vis_3d(emb, labels, f"figs/{dataname}_{alg}_3d.pdf")
        np.savetxt(f"embeddings/{dataname}_{alg}_3d.txt",emb)

    for alg, emb in zip(algs,embeddings):
        vis_2d(emb,labels,f"figs/{dataname}_{alg}_2d.pdf")
    
    for key in vals: 
        print(f"{key} -> {vals[key].keys()}")

    return vals

# from new_enstsne import load_clusters, load_penguins, load_auto, load_food, load_fashion, load_cc, load_wine, load_mnist
def find_enstsne():

    dists, labels, X = load_clusters(400, [20,20,20], [4,3,2])

    eps = lambda: np.random.normal(0,0.01)

    for d,lab in zip(dists,labels):
        for i in range(d.shape[0]):
            for j in range(i):
                if lab[i] == lab[j]: d[i,j] = 1 + eps()
                else: d[i,j] = 2 + eps()
                d[j,i] = d[i,j]

    print(dists[0])

    new_ens = ENSTSNE(dists, 18, labels, early_exaggeration=6)
    new_ens.gd(1000,lr=50)
    X = new_ens.get_embedding()
    subs = new_ens.get_images()

    vis_3d(X,labels,f"figs/ens-t-sne-3d-synthetic-4-3-2.pdf")
    vis_2d(subs,labels, f"figs/ens-t-sne-2d-synthetic-4-3-2.pdf")



from new_enstsne import load_clusters, load_penguins, load_auto, load_fashion, load_food, load_mnist, load_cc
if __name__ == "__main__":
    # find_enstsne()
    funcs = [
        ["synthetic", lambda : load_clusters(300, [10,10], [2,3])],
        ["penguins", lambda : load_penguins()],
        ["auto", lambda: load_auto()],
        ["food", lambda: load_food()],
        # ["fashion", lambda: load_fashion()],
        ["mnist", lambda: load_mnist()],
        # ["credit_card", lambda: load_cc()],
        # ["wine", lambda: load_wine()]
        ]
    
    for name, f in funcs:
        dists, labels, X = f()

        results = eval(dists,labels,pairwise_distances(X),X,name)

        lab = np.array(labels)
        np.savetxt(f"embeddings/labels_{name}.txt", lab.T)

        # results = {}

        import pickle 
        # open a file, where you ant to store the data
        file = open(f'results/{name}.pkl', 'wb')

        # dump information to that file
        pickle.dump(results, file)

        # close the file
        file.close()    


