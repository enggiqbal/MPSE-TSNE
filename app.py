import dash
import dash_bootstrap_components as dbc
external_stylesheets = [dbc.themes.CYBORG]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# app.css.append_css({'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'})
app.config.suppress_callback_exceptions = True
server = app.server
app.config.suppress_callback_exceptions = True