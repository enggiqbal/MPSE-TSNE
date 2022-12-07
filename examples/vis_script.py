import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.patheffects as PathEffects


draw_labels = True

X1 = np.loadtxt('mview/temp/images_0.csv')
X2 = np.loadtxt('mview/temp/images_1.csv')
# C1 = np.loadtxt('mview/temp/sample_classes.csv')[0,:]

data = pd.read_csv("food_comp_processed.csv")

X = data.drop(['Unnamed: 0', "Shrt_Desc"],axis=1).to_numpy()
from sklearn.cluster import KMeans
C1 = KMeans(3).fit_predict(X / np.max(X,axis=0))

keep_labels = [
    "TURKEY,WHL,MEAT ONLY,RAW",
    "INFANT FORMULA,MEAD JOHNSON,ENFAMIL,LO IRON,RTF",
    "MILK,RED FAT,FLUID,2% MILKFAT,W/ ADDED VIT A & VITAMIN D",
    "ALCOHOLIC BEV,WINE,TABLE,ALL",
    "BEEF,CHUCK FOR STEW,LN & FAT,ALL GRDS,RAW",
    "POTATOES,BKD,FLESH,W/SALT",
    "BEANS,KIDNEY,RED,MATURE SEEDS,CND,SOL & LIQUIDS", 
    "APPLES,RAW,WITH SKIN",
    "BANANAS,RAW"
]
in_eng = [
    'Turkey',
    'Infant formula',
    'Milk 2%',
    'Wine',
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

cmap = {0: "red", 
        1: "blue",
        2: "orange",
        3: "tab:red",
        4: "tab:purple"}

C = [cmap[c] for c in C1]
C_copy = [cmap[c] for c in C1]

data = pd.read_csv('food_comp_processed.csv')
labels = data[['Shrt_Desc']].to_numpy()

fig, (ax1,ax2) = plt.subplots(1,2)


scatter_dict = dict()
annot_dict = dict()

def hover_scatter(fig,ax,X,shrt_labels,labels):
        print(labels)
        scatter_dict[ax] = ax.scatter(X[:,0],X[:,1],s=20,c=C)
        N = X.shape[0]
        # if labels is True:
        #     labels = range(N)
        for i in range(len(shrt_labels)):
            x,y = X[i,0], X[i,1]
            if shrt_labels[i] == "Apple": y = y-0.2
            if shrt_labels[i] == "Infant formula": y = y+0.25
            if shrt_labels[i] == "Kidney Beans": y = y-0.1
            txt = ax.annotate(shrt_labels[i],(x,y),textcoords="offset points",
                        xytext=(0,4),ha='center',fontsize='large',weight="bold",
                        horizontalalignment='center',color='black')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])


        annot_dict[ax] = ax.annotate("", xy=(0,0), xytext=(0,0),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        annot_dict[ax].set_visible(False)

        def update_annot(ind,annot,scatter):
            pos = scatter.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            # text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
            #                     " ".join([names[n] for n in ind["ind"]]))
            annot.set_text(labels[ind["ind"][0]])
            print(labels[ind["ind"][0]])
            # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            # annot.get_bbox_patch().set_alpha(0.4)


        def hover(event):
            if event.inaxes in [ax1, ax2]:
                for ax in [ax1, ax2]:
                    cont, ind = scatter_dict[ax].contains(event)
                    if cont:
                        update_annot(ind, annot_dict[ax],scatter_dict[ax])
                        # C_copy[ind["ind"][0]] = "red"
                        # ax.collections[0].set_color(C_copy)
                        annot_dict[ax].set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if annot_dict[ax].get_visible():
                            # C_copy[ind['ind'][0]] = C[ind['ind'][0]]
                            # ax.collections[0].set_color(C_copy)
                            annot_dict[ax].set_visible(False)
                            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()

hover_scatter(fig,ax1,X1,shrt_labels,labels)
hover_scatter(fig,ax2,X2,shrt_labels,labels)

plt.show()

X = data.drop(['Unnamed: 0', "Shrt_Desc"],axis=1).to_numpy()
from sklearn.manifold import TSNE
Y = TSNE(perplexity=160,init='pca',learning_rate='auto').fit_transform(X / np.max(X,axis=0))
#Y = PCA(2).fit_transform(X / np.max(X,axis=0))

fig, ax = plt.subplots()
hover_scatter(fig,ax,Y,shrt_labels,labels)
# for i in range(Y.shape[0]):
#     ax.annotate(shrt_labels[i],(Y[i,0],Y[i,1]) if shrt_labels[i] != "Kidney Beans" else (Y[i,0],Y[i,1]+0.2),
#                         textcoords="offset points",
#                         xytext=(0,4),ha='center',fontsize='small',
#                         horizontalalignment='center',color='black')

ax.set_xticks([])
ax.set_yticks([])
plt.show()

fig= plt.figure(figsize=(5,4))
ax = fig.add_subplot(1,1,1,projection='3d')

X = np.loadtxt("mview/temp/embedding.csv")

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

N = len(X)

x,y,z = X[:,0], X[:,1], X[:,2]
for px,py,pz,c in zip(x,y,z, C1):
    ax.scatter3D(px,py,pz,c=cmap[c], alpha=0.9)

ax.grid(color='r')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.setp(ax.spines.values(), color='blue')
plt.show()
#What can we see that we can't see in small multiples?