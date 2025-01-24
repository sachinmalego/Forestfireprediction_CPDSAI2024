'''
* https://dash.gallery/dash-clinical-analytics/
'''

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pages.dashboard as dashboard
import pages.prediction as prediction
import dash_bootstrap_components as dbc


external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=New+Amsterdam&display=swap",
    dbc.themes.SUPERHERO, # https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/explorer/
    '/assets/style.css'
]

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=False, external_stylesheets=external_stylesheets)
app.title = 'Fire Risk Prediction'
server = app.server

# Define application layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Update page content based on URL
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/prediction':
        return prediction.layout
    else:
        return dashboard.layout

if __name__ == '__main__':
    app.run_server(debug=True)
