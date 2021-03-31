
import matplotlib.pyplot as plt

from dash.dependencies import Input, Output, State
from app import app
from layout import MPSELayout
from dataset import MPSEDataset
from dash import callback_context
import dash
from vissettings import VIS
import plotly.express as px
from dash.exceptions import PreventUpdate


import sys
sys.path.insert(0,'../') 
import mview 
import numpy as np 
 

class MPSETSNE:
    def __init__(self, inputwords):
        self.app = app
        self.dataset = None
        self.layout = MPSELayout(app).layout
        self.X = []
        self.MPSEDataset=MPSEDataset()
        
        w1=inputwords.split(",")[0].strip() 
        w2=inputwords.split(",")[1].strip() 
        D,labels=self.MPSEDataset.get_distance_matrics(w1, w2,50)
        self.get_mview(D,labels)

    def get2Dproject(self, points, P):
            Z=np.min( np.sum(P,axis=1))
            A= np.sum(P,axis=1) !=Z
            svd=np.linalg.svd(P)
            proj=svd[2]
            proj=proj[0:2,:]
            proj = proj.transpose()
            return np.matmul(points, proj)
    

    def get_mview(self,D ,label):
            mv = mview.mpse_tsne(D, verbose=0,   show_plots=False)
            self.X = mv.embedding
            self.Q = mv.projections
            self.nprojections=len(mv.individual_cost)
            self.cost = [0,0]
            self.info=f"Proj1: {mv.individual_cost[0]:0.3f}, Proj2: {mv.individual_cost[1]:0.3f}"
            if self.nprojections==3:
               self.info=self.info + f"Proj3: {mv.individual_cost[2]:0.3f} "
            self.cost_fig = px.line(
                x=range(0, len(self.cost)), y=self.cost,  template='plotly_dark' )
            #temp save picture 
            ps1=self.get2Dproject(self.X, self.Q[0])
            ps2=self.get2Dproject(self.X, self.Q[1])

            figx, ax = plt.subplots()
            #color=label["source"]
            ax.scatter(ps1.T[0], ps1.T[1])
            for i, txt in enumerate(label["allwords"]):
                ax.annotate(txt, (ps1[i][0], ps1[i][1]))
            plt.axis('off')
            figx.savefig("test.png")

            ps1=ps2
            figx, ax = plt.subplots()
            ax.scatter(ps1.T[0], ps1.T[1])
            for i, txt in enumerate(label["allwords"]):
                ax.annotate(txt, (ps1[i][0], ps1[i][1]))
            plt.axis('off')
            figx.savefig("test2.png")

             

if __name__ == '__main__':
    inputwords="computer,tv"
    a = MPSETSNE(inputwords)
    
