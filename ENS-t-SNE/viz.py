import matplotlib.pyplot as plt
import numpy as np
import pickle 

file = open('test.pkl', 'rb')
data = pickle.load(file)
file.close()

for view,algs in data.items():

    species = algs["enstsne"].keys()
    penguin_means = {
        'Bill Depth': (18.35, 18.43, 14.98),
        'Bill Length': (38.79, 48.83, 47.50),
        'Flipper Length': (189.95, 195.82, 217.19),
    }
    penguin_means = { "enstsne": algs["enstsne"].values(), 
                      "tsne"   : algs["tsne"].values(), 
                      "umap"   : algs["umap"].values()}

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3,fmt="%.3f")
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Length (mm)')
    ax.set_title(f'{view} food data')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 2)

    plt.show()