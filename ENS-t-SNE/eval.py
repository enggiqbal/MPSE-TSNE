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

def plot_3d(Emb: np.array, C:np.array,title=None):
    #Plot 3D
    fig= plt.figure(figsize=(5,4))
    ax = fig.add_subplot(1,1,1,projection='3d')

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

    # N = len(Emb)

    x,y,z = Emb[:,0], Emb[:,1], Emb[:,2]
    ax.scatter3D(x,y,z,c=[tab10[c] for c in C],alpha=0.9)

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


def plot2d(y1,y2,C,title=None):
    #Plot results
    fig, (ax1,ax2) = plt.subplots(1,2)
    # y1 = mv.images[0,:,:]
    # y2 = mv.images[1,:,:]
    C = [tab10[c] for c in C]
    for ax,emb in zip([ax1,ax2],[y1,y2]):
        x , y = emb[:,0], emb[:,1]
        ax.scatter(x,y,c=C)
        for i in range(len(labels)):
            x,y = emb[i,0], emb[i,1]
            txt = ax.annotate(labels[i],(x,y),textcoords="offset points",
                        xytext=(0,4),ha='center',fontsize='large',weight="bold",
                        horizontalalignment='center',color='black')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.set_size_inches(8,8)    
    if title: plt.savefig(title)
    else: plt.show()

if __name__ == "__main__":

    #Load food comp data into pandas df 
    data = pd.read_csv("datasets/food_comp/food_comp_processed.csv")

    #Parse data
    #First subspace; water+lipids
    x1 = data[['Water_(g)','Vit_E_(mg)','Sodium_(mg)','Lipid_Tot_(g)','Energ_Kcal']].to_numpy()

    #second subspace; proteins+B vitamins
    x2 = data[['Protein_(g)', 'Vit_B6_(mg)', 'Vit_B12_(µg)', 'Vit_D_µg']].to_numpy()

    #All dimensions present in dataset
    X = data.drop(['Unnamed: 0', "Shrt_Desc"],axis=1).to_numpy()


    #Cluster High dimensional data
    from sklearn.cluster import KMeans
    C = KMeans(3).fit_predict(X / np.max(X,axis=0))

    #Parse data labels
    labels = [str(lbl) for lbl in data[['Shrt_Desc']].values.tolist()]
    eng_map = dict(zip(keep_labels, in_eng))
    labels = [eng_map[lbl] if lbl in keep_labels else "" for lbl in data[['Shrt_Desc']].to_numpy().reshape(-1)]
    

    #Normalize subspaces 
    x1 /= np.max(x1,axis=0)
    x2 /= np.max(x2,axis=0)

    #Choose initial orthogonal projections
    proj1 = np.array([[1,0,0],
                      [0,1,0]])

    proj2 = np.array([[0,1,0],
                      [0,0,1]])                  

    #Call ENS-t-SNE module
    mv = mview.mpse_tsne([x1,x2],perplexity=500,
                    iters=500, smart_init=True,
                    show_plots=False,initial_projections=[proj1,proj2])


    y1 = mv.images[0,:,:]
    y2 = mv.images[1,:,:]
    plot2d(y1,y2,C,title="figs/enstsne2d.png")
    plt.clf()
    # fig.tight_layout()


    Emb = mv.embedding
    plot_3d(Emb,C,title="figs/enstsne3d.png")

    
    from metrics import compute_all_metrics

    x3 =  data[['Water_(g)','Vit_E_(mg)','Sodium_(mg)','Lipid_Tot_(g)','Energ_Kcal','Protein_(g)', 'Vit_B6_(mg)', 'Vit_B12_(µg)', 'Vit_D_µg']].to_numpy()

    u3d = UMAP(n_components=3).fit_transform(x3)
    plot_3d(u3d,C,title="figs/umap3d.png")
    t3d = TSNE(n_components=3).fit_transform(x3)
    plot_3d(t3d,C,title="figs/tsne3d.png")

    three_stats = {
        "enstsne": compute_all_metrics(Emb,x3,C),
        "tsne":    compute_all_metrics(t3d,x3,C),
        "umap":    compute_all_metrics(u3d,x3,C)
    }

    u1 = UMAP().fit_transform(x1)
    u2 = UMAP().fit_transform(x2)
    plot2d(u1,u2,C,title="figs/umap2d.png")
    t1 = TSNE().fit_transform(x1)
    t2 = TSNE().fit_transform(x2)
    plot2d(t1,t2,C,title="figs/tsne2d.png")

    view1 = {
        "enstsne": compute_all_metrics(y1,x1,C),
        "tsne":    compute_all_metrics(t1,x1,C),
        "umap":    compute_all_metrics(u1,x1,C)        
    }

    view2 = {
        "enstsne": compute_all_metrics(y2,x2,C),
        "tsne":    compute_all_metrics(t2,x2,C),
        "umap":    compute_all_metrics(u2,x2,C)        
    }    

    import pickle 
    # open a file, where you ant to store the data
    file = open('test.pkl', 'wb')

    # dump information to that file
    pickle.dump({"view1": view1, "view2": view2, "3d": three_stats}, file)

    # close the file
    file.close()    