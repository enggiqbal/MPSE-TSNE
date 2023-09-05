import matplotlib.pyplot as plt
import numpy as np
import pickle 
import os 

cmap = plt.get_cmap("Set2")

for fname in os.listdir("results/"):
    print(fname)

    file = open(f"results/{fname}", 'rb')
    data = pickle.load(file)
    file.close()

    for view,algs in data.items():

        species = algs["ens-t-sne"].keys()
        penguin_means = { 
                        # "old-ens-t-sne": algs["old_ens"].values(),
                        "ens-t-sne": algs["ens-t-sne"].values(),
                        "mds"      : algs["mds"].values(), 
                        "tsne"   : algs["tsne"].values(), 
                        "umap"   : algs["umap"].values(),
                        "mpse"   : algs["mpse"].values()
                        }

        x = np.arange(len(species))  
        width = 1 / (len(penguin_means) + 1) 
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for i, (attribute, measurement) in enumerate(penguin_means.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, color=cmap(i), label=attribute)
            ax.bar_label(rects, padding=3,fmt="%.3f")
            multiplier += 1

        ax.set_title(f"{view} {fname.split('.')[0]} data")
        ax.set_xticks(x + width, [s if s != "cluster variance" else "CEV" for s in species])
        if fname.split(".")[0] == "penguins":
            ax.legend(loc='upper left')
        # ax.set_ylim(0, 2)

        plt.savefig(f"figs/{fname.split('.')[0]}-{view}.png")