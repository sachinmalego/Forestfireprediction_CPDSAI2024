from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pickle
import numpy as np
from app import app
import os

# Load the saved model
model_path = os.path.join(os.path.dirname(__file__), '../model/best_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define layout for the prediction page
layout = html.Div([
    html.H1('Fire Risk Prediction', className='page-title'),
    dcc.Link('Go to Dashboard Page', href='/', className='link-button'),
    html.Div([
        
        html.Div([
            html.Label('Temperature (°C)', className='input-label'),
            dcc.Input(id='input-temperature', type='number', value=30, className='input-field'),
            html.Label('Humidity (%)', className='input-label'),
            dcc.Input(id='input-humidity', type='number', value=50, className='input-field'),
        ]),
        
        html.Div([
            html.Label('Wind Speed (km/h)', className='input-label'),
            dcc.Input(id='input-wind-speed', type='number', value=10, className='input-field'),
            html.Label('Rain (mm)', className='input-label'),
            dcc.Input(id='input-rain', type='number', value=0, className='input-field'),
        ]),
        
        html.Div([
            html.Label('FFMC', className='input-label'),
            dcc.Input(id='input-ffmc', type='number', value=85, className='input-field'),
            html.Label('DMC', className='input-label'),
            dcc.Input(id='input-dmc', type='number', value=60, className='input-field'),
        ]),
        
        html.Div([
            html.Label('DC', className='input-label'),
            dcc.Input(id='input-dc', type='number', value=300, className='input-field'),
            html.Label('ISI', className='input-label'),
            dcc.Input(id='input-isi', type='number', value=10, className='input-field'),
        ]),
        
        html.Label('Sea Surface Temperature (Nino 3.4 Index)'),
        dcc.Input(id='input-nino34', type='number', value=0.5, className='input-field'),
        dbc.Button(id='submit-button', type='submit', children='Predict Fire Risk', className='submit-button me-1', n_clicks=0, color='danger')
    ], className='input-container'),
    html.Div(id='prediction-output', className='prediction-output', children='')
], className='prediction-page')

# Define callback for making prediction
@app.callback(
    Output('prediction-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('input-temperature', 'value'),
     State('input-humidity', 'value'),
     State('input-wind-speed', 'value'),
     State('input-rain', 'value'),
     State('input-ffmc', 'value'),
     State('input-dmc', 'value'),
     State('input-dc', 'value'),
     State('input-isi', 'value'),
     State('input-nino34', 'value')]
)
def predict_fire_risk(n_clicks, temperature, humidity, wind_speed, rain, ffmc, dmc, dc, isi, nino34_sst):
    if n_clicks and n_clicks > 0:
        # Generate a dummy fire risk percentage
        prediction = np.random.uniform(0, 100)  # Random fire risk in percentage - just for demonstration
        return f'Chance of Fire: {prediction:.2f}%'
    return ''