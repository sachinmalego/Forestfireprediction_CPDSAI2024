import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Create Dash application instance
app = dash.Dash(__name__)

# Generate dummy data for fire risk in June 2012
np.random.seed(42)  # For reproducibility

days = list(range(1, 31))  # Days of June
fire_risk = np.random.uniform(0, 100, size=len(days))  # Random fire risk percentage

# Create DataFrame
fire_data = pd.DataFrame({
    'day': days,
    'fire_risk': fire_risk
})

# Reshape data to display 7 days per row
fire_data['week'] = (fire_data['day'] - 1) // 7 + 1
fire_data['weekday'] = (fire_data['day'] - 1) % 7 + 1

# Pivot the data to create a heatmap layout with 7 days per row
heatmap_data = fire_data.pivot(index='week', columns='weekday', values='fire_risk')

# Create heatmap using Plotly with Superhero theme adjustments
heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_data.values, # Heatmap values
    x=heatmap_data.columns, # Weekdays
    y=['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5'],
    colorscale='Reds',
    hoverongaps=False, # Display empty days as empty cells
    hovertemplate='%{x} June 2012<br>Fire Risk: %{z:.2f}%<extra></extra>'  # Custom hover template
))
heatmap.update_layout(
    title='Fire Risk Heatmap - June 2012',
    xaxis=dict(side='top', showgrid=False),  # Place x-axis labels at the top and hide grid
    yaxis=dict(autorange='reversed', showgrid=False),  # Display first week at the top and hide grid
    paper_bgcolor='#2b3e50',
    plot_bgcolor='#2b3e50',
    font=dict(color='#f2f5f7'),
    title_font=dict(size=24, color='#f2f5f7'),
    coloraxis_colorbar=dict(
        title='Fire Risk (%)'
    )
)

# Define layout for the dashboard page
layout = html.Div([
    html.H1('Fire Risk Dashboard', className='page-title'),
    dcc.Link('Go to Prediction Page', href='/prediction', className='link-button'),
    html.Div([
        dcc.Graph(figure=heatmap)
    ], style={'width': '800px', 'margin': '0 auto'})
])

# Callback to update text based on heatmap cell click
@app.callback(
    dash.dependencies.Output('click-data', 'children'),
    [dash.dependencies.Input('heatmap', 'clickData')]
)
def display_click_data(clickData):
    if clickData is None:
        return "Click on a cell in the heatmap to see details here."
    point = clickData['points'][0]
    return f"Week: {point['y']}, Day: {point['x']} June 2012, Fire Risk: {point['z']:.2f}%"

# Add a Div to display click data
layout.children.append(html.Div(id='click-data', style={'marginTop': '20px', 'color': '#f2f5f7'}))