import dash
from dash import dcc, html, Input, Output, State, callback
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import dash_bootstrap_components as dbc
import time
import joblib

# Load dataset
file_path = 'data/forest_fires_merged_with_coords.csv'
data = pd.read_csv(file_path)

# Load the trained model
model_path = 'best_model.pkl'
rf_model = joblib.load(model_path)

# Define the layout for the prediction page
def get_prediction_layout():
    return dbc.Container([
        dbc.Row([
            # Left Column (Input Form)
            dbc.Col([
                dbc.Card(
                    [
                        html.H2('Fire Prediction', style={'textAlign': 'center', 'color': '#008000', 'padding-top':'10px'}),
                        html.Div([
                            # First Row of Inputs
                            dbc.Row([
                                dbc.Col([
                                    html.Label('Temperature (°C):', style={'font-weight': 'bold'}),
                                    dcc.Input(id='input-temp', type='number', placeholder='Enter temperature', 
                                            className='form-control', style={'border-radius': '15px', 'padding': '12px'}),
                                ], width=6),
                                
                                dbc.Col([
                                    html.Label('Humidity (%):', style={'font-weight': 'bold'}),
                                    dcc.Input(id='input-humidity', type='number', placeholder='Enter humidity', 
                                            className='form-control', style={'border-radius': '15px', 'padding': '12px'}),
                                ], width=6)
                            ], style={'margin-bottom': '15px'}),

                            # Second Row of Inputs
                            dbc.Row([
                                dbc.Col([
                                    html.Label('Wind (km/h):', style={'font-weight': 'bold'}),
                                    dcc.Input(id='input-wind', type='number', placeholder='Enter wind speed', 
                                            className='form-control', style={'border-radius': '15px', 'padding': '12px'}),
                                ], width=6),
                                
                                dbc.Col([
                                    html.Label('Rain (mm):', style={'font-weight': 'bold'}),
                                    dcc.Input(id='input-rain', type='number', placeholder='Enter rainfall', 
                                            className='form-control', style={'border-radius': '15px', 'padding': '12px'}),
                                ], width=6)
                            ], style={'margin-bottom': '15px'}),

                            # Third Row of Inputs
                            dbc.Row([
                                dbc.Col([
                                    html.Label('FFMC:', style={'font-weight': 'bold'}),
                                    dcc.Input(id='input-FFMC', type='number', placeholder='Enter FFMC', 
                                            className='form-control', style={'border-radius': '15px', 'padding': '12px'}),
                                ], width=6),
                                
                                dbc.Col([
                                    html.Label('DMC:', style={'font-weight': 'bold'}),
                                    dcc.Input(id='input-DMC', type='number', placeholder='Enter DMC', 
                                            className='form-control', style={'border-radius': '15px', 'padding': '12px'}),
                                ], width=6)
                            ], style={'margin-bottom': '15px'}),

                            # Fourth Row of Inputs
                            dbc.Row([
                                dbc.Col([
                                    html.Label('DC:', style={'font-weight': 'bold'}),
                                    dcc.Input(id='input-DC', type='number', placeholder='Enter DC', 
                                            className='form-control', style={'border-radius': '15px', 'padding': '12px'}),
                                ], width=6),
                                
                                dbc.Col([
                                    html.Label('ISI:', style={'font-weight': 'bold'}),
                                    dcc.Input(id='input-ISI', type='number', placeholder='Enter ISI', 
                                            className='form-control', style={'border-radius': '15px', 'padding': '12px'}),
                                ], width=6)
                            ], style={'margin-bottom': '15px'}),

                            # Fifth Row of Inputs
                            dbc.Row([
                                dbc.Col([
                                    html.Label('NINO3.4 SST:', style={'font-weight': 'bold'}),
                                    dcc.Input(id='input-nino34', type='number', placeholder='Enter NINO3.4 SST', 
                                            className='form-control', style={'border-radius': '15px', 'padding': '12px'}),
                                ], width=6)
                            ], style={'margin-bottom': '25px'}),
                            
                            # Predict Button
                            html.Button('Predict Fire Risk', id='predict-btn', n_clicks=0, style={
                                'width': '100%', 'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 
                                'padding': '15px', 'border-radius': '25px', 'font-size': '16px', 'cursor': 'pointer',
                                'box-shadow': '0 4px 6px rgba(0,0,0,0.1)', 'transition': 'all 0.3s ease'
                            })
                    ], style={'background-color': '#f8f9fa', 'padding': '30px', 'border-radius': '10px'}),
                ])
            ], width=6),

            # Right Column (Prediction Output and Gauge)
            dbc.Col([
                dbc.Card([
                    html.H3('Prediction Result', style={'textAlign': 'center', 'margin-bottom': '20px', 'color': '#008000', 'padding-top':'10px'}),
                    # Empty initial state for gauge chart
                    dcc.Graph(id='fire-gauge', figure={
                        'data': [],
                        'layout': go.Layout(
                            title='Fire Probability Gauge',
                            showlegend=False,
                            geo=dict(showland=True),
                            font=dict(size=12),
                            height=300
                        )
                    }),
                    html.Div(id='prediction-output', style={'font-size': '24px', 'textAlign': 'center'}),
                ])
            ], width=6)
        ], style={'margin-top': '50px'})
    ])

# Callback for Prediction
@callback(
    [Output('prediction-output', 'children'),
     Output('fire-gauge', 'figure')],
    [Input('predict-btn', 'n_clicks')],
    [State('input-temp', 'value'),
     State('input-humidity', 'value'),
     State('input-wind', 'value'),
     State('input-rain', 'value'),
     State('input-FFMC', 'value'),
     State('input-DMC', 'value'),
     State('input-DC', 'value'),
     State('input-ISI', 'value'),
     State('input-nino34', 'value')]
)
def predict_fire(n_clicks, temp, humidity, wind, rain, FFMC, DMC, DC, ISI, nino34_sst):
    if n_clicks > 0:
        # Fill missing inputs with the latest available data
        input_data = {
            'temp': temp or data['temp'].iloc[-1],  # Use last available if blank
            'humidity': humidity or data['humidity'].iloc[-1],
            'wind': wind or data['wind'].iloc[-1],
            'rain': rain or data['rain'].iloc[-1],
            'FFMC': FFMC or data['FFMC'].iloc[-1],
            'DMC': DMC or data['DMC'].iloc[-1],
            'DC': DC or data['DC'].iloc[-1],
            'ISI': ISI or data['ISI'].iloc[-1],
            'nino34_sst': nino34_sst or data['nino34_sst'].iloc[-1],
        }
        
        # Prediction
        prediction = rf_model.predict([list(input_data.values())])
        probability = rf_model.predict_proba([list(input_data.values())])[0][1]
        
        # Fire prediction output text
        prediction_text = f'Prediction: {"Fire" if prediction[0] == 1 else "No Fire"} | Probability: {probability*100:.2f}%'
        
        # Create gauge chart for fire prediction probability
        gauge_figure = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Probability of Fire Occurrence", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [None, 100]},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': probability * 100
                }
            },
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
        
        # Animate the gauge by adding a small delay before showing the prediction
        time.sleep(0.5)  # Wait for 0.5 seconds before updating gauge
        
        return prediction_text, gauge_figure
    # Return initial empty state for the gauge if no prediction is made yet
    return '', go.Figure({
        'data': [],
        'layout': go.Layout(
            title='Fire Probability Gauge',
            showlegend=False,
            geo=dict(showland=True),
            font=dict(size=12),
            height=300
        )
    })

