import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
from home_page import get_home_layout  # Import home page layout
from prediction_page import get_prediction_layout  # Import prediction page layout

# Create the Dash app and use Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create Navbar with Bootstrap theme
def create_navbar():
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Forest Fire Prediction - CPDSAI2024", href="/", style={'font-size': '24px', 'font-weight': 'bold', 'color': 'white'}),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Dashboard", href="/", active="exact", style={'font-size': '18px', 'font-weight': '500'})),
                    dbc.NavItem(dbc.NavLink("Prediction", href="/prediction", active="exact", style={'font-size': '18px', 'font-weight': '500'})),
                ],
                navbar=True,
                style={'text-align': 'center'}
            ),
        ]),
        color="dark",
        dark=True,
        style={'border-bottom': '2px solid #ff4d4d', 'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)'}
    )

# App Layout
app.layout = html.Div([
    create_navbar(),  # Add the navbar here
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Page content routing
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/prediction':
        return get_prediction_layout()
    else:
        return get_home_layout()

if __name__ == '__main__':
    app.run_server(debug=False,dev_tools_ui=False,dev_tools_props_check=False)
