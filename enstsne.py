from ENSTSNE import new_enstsne
import pylab as plt

if __name__ == "__main__":
    dists, labels, X = new_enstsne.load_penguins()

    enstsne = new_enstsne.ENSTSNE(dists,30,labels,fixed=False)
    enstsne.gd(1000,0.5,50)

    enstsne.vis_3d()