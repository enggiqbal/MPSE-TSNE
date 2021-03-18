import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import os

class MPSELayout:
    def __init__(self,app):
        self.app=app
        self.layout = html.Div([
            html.H1("MPSE - TSNE",
                    style={'text-align': 'center', 'font-size': '250%'}),
            dbc.Row([
                    dbc.Col(dbc.FormGroup([ dbc.Label("Select dataset", html_for="dataset-dropdown"), self.datasetDropdown()]),width=1),
                    dbc.Col(dbc.FormGroup([ dbc.Label("Select projection", html_for="projection"), self.projectionDropdown()]),width=1),
                    dbc.Col(dbc.FormGroup([ dbc.Label("Perplexity", html_for="perplexity"), self.perplexity()]),width=1),
                     dbc.Col(dbc.FormGroup([ dbc.Label("Max Iter", html_for="max_iter"), self.max_iters()]),width=1),
                      dbc.Col(dbc.FormGroup([ dbc.Label("Smart Initialization", html_for="smart_initialization"), self.smart_initialization()]),width=2),
                      dbc.Col(dbc.FormGroup([ dbc.Label("Algorithm", html_for="visualization_method"),  self.visualization_method()]),width=1),
                    dbc.Col(dbc.FormGroup([ dbc.Label("", html_for="run-button"),  dbc.Button('Run',  color="primary", 
                                       id='run-button', className="mr-2", n_clicks=0)]),width=1)
                    ], justify="center", className="h-50"),
            dbc.Row([
                    dbc.Col([
                        html.Div(id='dd-output-container'),
                        html.Div(id='intermediate-value',
                                 style={'display': 'none'})
                    ] ),
                    ], style= {"marginBottom": 8 }), 
            dbc.Row(self.eyes(), justify="center", style= {"marginBottom": 8 }),
            dbc.Row(
            [   
                
      
            dbc.Col(self.loading("loading1" ,dcc.Graph( id='cost', style={  'height': '80vh'}, figure={})), width=3),
            dbc.Col(self.loading( "loading2" , dcc.Graph( id='main3d', style={ 'height': '80vh'}, figure={})), width=9)
           
            ]
               )

        ])
    def loading(self,loading_id, id):
        return  dcc.Loading(
            id=loading_id,
            type="default",
            children=id
        )

    def eyes(self):
        return dbc.Col([  
            dbc.Button('View 1',  color="primary",    id='proj1', className="mr-2", n_clicks=0),
             dbc.Button('View 2',  color="primary",    id='proj2', className="mr-2", n_clicks=0),
              dbc.Button('View 3',  color="primary",    id='proj3', className="mr-2", n_clicks=0),

            # html.Img(id="proj1x",   src= self.app.get_asset_url("eye.png")),
            #  html.Img(id="proj2x",src= self.app.get_asset_url("eye.png")), 
            #  html.Img(id="proj3x",src= self.app.get_asset_url("eye.png"))
             
             ], width=3)
             
    def datasetDropdown(self):
        return dcc.Dropdown(
            id='dataset-dropdown',
            options=[
                {'label': 'Credit Card', 'value': 'credit'},
                {'label': 'The New York Times News', 'value': 'newyork'},
            ],
            value='credit',
            searchable=False, placeholder="Select a dataset",
        )
    def max_iters(self):
        return dbc.Input(id='max_iters', value=20, type="number")
    def perplexity(self):
        return dbc.Input(id='perplexity', value=200, type="number")
    def visualization_method(self):
        return dcc.Dropdown(
            id='visualization_method',
            options=[{'label': 'MPSE-TSNE', 'value': 'tsne'},
                     {'label': 'MPSE-MDS', 'value': 'mds'}],
                     value='tsne',
            searchable=False, placeholder="Select algorithm type")
    def smart_initialization(self):

        return dcc.Dropdown(
            id='smart_initialization',
            options=[{'label': 'random', 'value': 'random'},
                     {'label': 'MDS-Based', 'value': 'true'}],
                     value='random',
            searchable=False, placeholder="Select smart initialization")

    def projectionDropdown(self):
        return dcc.Dropdown(
            id='projection',
            options=[{'label': 'fixed', 'value': 'standard'},
                     {'label': 'variable', 'value': 'variable'}],
                     value='standard',
            searchable=False, placeholder="Select project type")
