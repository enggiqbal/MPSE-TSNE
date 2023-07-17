import matplotlib.pyplot as plt
import numpy as np
import pickle 
import os 

for fname in os.listdir("results/"):
    print(fname)

    file = open(f"results/{fname}", 'rb')
    data = pickle.load(file)
    file.close()

    for view,algs in data.items():

        species = algs["ens-t-sne"].keys()
        penguin_means = { 
                        "old-ens-t-sne": algs["old_ens"].values(),
                        "ens-t-sne": algs["ens-t-sne"].values(),
                        "mds"      : algs["mds"].values(), 
                        "tsne"   : algs["tsne"].values(), 
                        "umap"   : algs["umap"].values(),
                        "mpse"   : algs["mpse"].values()}

        x = np.arange(len(species))  # the label locations
        width = 1 / (len(penguin_means) + 1) # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for attribute, measurement in penguin_means.items():
            offset = width * multiplier
            measurement = list(measurement)
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3,fmt="%.3f")
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        # ax.set_ylabel('Length (mm)')
        ax.set_title(f"{view} {fname.split('.')[0]} data")
        ax.set_xticks(x + width, species)
        ax.legend(loc='upper left')
        # ax.set_ylim(0, 2)

        plt.savefig(f"figs/with-old-{fname.split('.')[0]}-{view}.png")