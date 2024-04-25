import matplotlib.pyplot as plt
import numpy as np
import pickle 
import os 

cmap = plt.get_cmap("Set2")

def load_file(s):
    with open(f"results/{s}", "rb") as fdata:
        data = pickle.load(fdata)
    return data

def table(fstring):
    data = load_file(fstring)

    s = ""
    up = "$\\uparrow$"
    down = "$\\downarrow$"
    for metric in data["3d"]["ens-t-sne"]:
            s += f"{metric} {up if metric != 'stress' else down} & "
            for view in ["3d", "view-0", "view-1"]:
                ens = data[view]["ens-t-sne"][metric]
                mpse = data[view]["mpse"][metric]
                if ens == mpse:
                    s += f"{ens:.5f} & {mpse:.5f}"
                elif metric != "stress" and ens > mpse: 
                    s += f"\\textbf{{{ens:.5f}}} & {mpse:.5f}"
                elif metric != "stress":
                    s += f"{ens:.5f} & \\textbf{{{mpse:.5f}}}"
                elif ens > mpse:
                    s += f"{ens:.5f} & \\textbf{{{mpse:.5f}}}"
                else:
                    s += f"\\textbf{{{ens:.5f}}} & {mpse:.5f}"
                if view != "view-1": s += " & "
            s += "\\\\ \\hline \n"
    print(s)

def table_all(fstring):
    data = load_file(fstring)
    algs = data["3d"].keys()

    fmt = lambda n,b: f"{n:.5f}" if not b else f"\\textbf{{{n:.5f}}}"

    s = ""
    up = "$\\uparrow$"
    down = "$\\downarrow$"
    s += " & " + " & ".join(algs) + "\\\\ \\hline \n"
    for metric in data["3d"]["ens-t-sne"]:
            s += f"{metric} {up if metric != 'stress' else down} & "
            view = "3d"
            nums = [data[view][alg][metric] for alg in algs]
            ind = np.argmax(nums) if metric != "stress" else np.argmin(nums)
            s += " & ".join(fmt(n,i == ind) for i,n in enumerate(nums))


            s += "\\\\ \\hline \n"
    print(s)

def bar_charts():
    for fname in os.listdir("results/"):
        print(fname)

        file = open(f"results/{fname}", 'rb')
        data = pickle.load(file)
        file.close()

        for view,algs in data.items():
            for alg in algs:
                del algs[alg]["stress"]

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
                print(i)
                offset = width * multiplier
                rects = ax.bar(x + offset, measurement, width, color=cmap(i), label=attribute)
                # ax.bar_label(rects, padding=3,fmt="%.3f")
                multiplier += 1

            ax.set_title(f"{view} {fname.split('.')[0]} data")
            ax.set_xticks(x + width, [s if s != "cluster variance" else "CEV" for s in species])
            if fname.split(".")[0] == "penguins":
                ax.legend(loc='upper left')
            # ax.set_ylim(0, 2)
            plt.legend()
            plt.savefig(f"figs/{fname.split('.')[0]}-{view}.png")


if __name__ == "__main__":
    table("penguins.pkl")
    print("----")
    table("auto.pkl")
    print("----")
    table("food.pkl")
    bar_charts()

    table_all("penguins.pkl")
    print("----")
    table_all("auto.pkl")
    print("----")
    table_all("food.pkl")    