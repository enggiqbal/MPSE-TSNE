
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

app.title = 'MPSE - TSNE'

class MPSETSNE:
    def __init__(self, app):
        self.app = app
        self.dataset = None
        self.layout = MPSELayout(app).layout
        self.X = []
        self.MPSEDataset=MPSEDataset()

    def run_server(self, port=8084):
        if self.app is not None and hasattr(self, 'callbacks'):
            self.callbacks(self.app)
        self.app.layout = self.layout
        self.app.run_server(port=port, host="0.0.0.0",debug=True)

    def callbacks(self, app):
        stateinput=['dataset-dropdown','perplexity','projection','visualization_method','max_iters','smart_initialization','inputwords']
        states=[State(s, 'value') for s in stateinput]
        @app.callback(
            [dash.dependencies.Output('dd-output-container', 'children'),
             dash.dependencies.Output('main3d', 'figure'),
             dash.dependencies.Output('cost', 'figure')],
            [
                dash.dependencies.Input('run-button', 'n_clicks'),
                dash.dependencies.Input('proj1', 'n_clicks'),
                dash.dependencies.Input('proj2', 'n_clicks'),
                dash.dependencies.Input('proj3', 'n_clicks'),
            ], states
        )
        def run(n_clicks, proj1, proj2, proj3, dataset, perplexity, projection,visualization_method_name,max_iters,smart_initialization,inputwords):
            trigger = callback_context.triggered[0]
            if trigger["prop_id"] == ".":
                raise PreventUpdate

            if 'proj' in trigger["prop_id"]:
                if len(self.X) == 0:
                    return ["Please run first.", {}, {}]
                clicked=int(trigger["prop_id"].replace("proj", "").replace(".n_clicks", "")) - 1
                if self.nprojections==2 and clicked==2:
                    raise PreventUpdate
                camera = dict(eye=self.get_viewpoint_from_projection(  clicked  ) )
                self.mainfig = self.mainfig.update_layout(scene_camera=camera)

            if smart_initialization!='True':
                smart_initialization=None 


            if dataset == "credit" and trigger["prop_id"] == "run-button.n_clicks":
                D, labels = self.MPSEDataset.read_credit_card_data()
                self.get_mview(projection,D,perplexity,max_iters,visualization_method_name,smart_initialization)
                self.mainfig = self.get_chart_fig(self.X, labels)
 
            if dataset=='news' and   trigger["prop_id"] == "run-button.n_clicks":
                w1=inputwords.split(",")[0].strip() 
                w2=inputwords.split(",")[1].strip() 
                D,labels=self.MPSEDataset.get_distance_matrics(w1, w2,50)
                if D==None: return [labels , {},{}] 
                self.get_mview(projection,D,perplexity,max_iters,visualization_method_name,smart_initialization)
                self.mainfig = self.get_chart_fig_news( labels)
            return [self.info, self.mainfig, self.cost_fig]

    def get_mview(self,projection,D,perplexity,max_iters,visualization_method_name,smart_initialization):
            if projection==None:
                mv = mview.basic(D,  verbose=2, visualization_args={'perplexity': perplexity},  max_iter=max_iters, smart_initialization=smart_initialization, visualization_method = visualization_method_name)
            else:
                mv = mview.basic(D, fixed_projections=projection, verbose=2, visualization_args={'perplexity': perplexity},  max_iter=max_iters, smart_initialization=smart_initialization, visualization_method = visualization_method_name)
            self.X = mv.X
            self.Q = mv.Q
            self.nprojections=len(mv.individual_cost)
            self.cost = mv.H[0]['costs']
            self.info=f"Proj1: {mv.individual_cost[0]:0.3f}, Proj2: {mv.individual_cost[1]:0.3f}"
            if self.nprojections==3:
               self.info=self.info + f"Proj3: {mv.individual_cost[2]:0.3f} "
            self.cost_fig = px.line(
                x=range(0, len(self.cost)), y=self.cost,  template='plotly_dark' )
            

    def get_viewpoint_from_projection(self, i):
        constant =3 #ZOOM
        a = self.Q[i][0]
        b = self.Q[i][1]
        p = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] -
             a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        return dict(x=p[0] * constant, y=p[1] * constant, z=p[2] * constant)

    def get_chart_fig_news(self, label):
        camera = dict(eye=self.get_viewpoint_from_projection(1))
        fig = px.scatter_3d( x=self.X.T[0], y=self.X.T[1], z=self.X.T[2], text=label["allwords"], size_max=10,color=label["source"], template='plotly_dark'  )
        fig.update_layout(scene_camera=camera)
        fig.update_layout(scene=VIS().scene)
        return fig


    def get_chart_fig(self, X, df):
        df["x"], df["y"], df["Z"] = (X.T[0] , X.T[1] ,  X.T[2])
        edu_order = {1: 'Lower secondary', 2: 'Secondary / secondary special',
                     3: 'Incomplete higher', 4: 'Higher education'}
        for x in edu_order:
            df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].replace(
                x, edu_order[x])

        camera = dict(eye=self.get_viewpoint_from_projection(1))
        df['CODE_GENDER'] = df['CODE_GENDER'].replace(2, "Male")
        df['CODE_GENDER'] = df['CODE_GENDER'].replace(1, "Female")
        fig = px.scatter_3d(df, x='x', y='y', z='z', symbol='CODE_GENDER',
                            size_max=40, color='NAME_EDUCATION_TYPE', size='AMT_INCOME_TOTAL', template='plotly_dark' )
        fig.update_layout(scene_camera=camera)
        fig.update_layout(scene=VIS().scene)
        return fig
 

if __name__ == '__main__':

    a = MPSETSNE(app)
    a.run_server()
