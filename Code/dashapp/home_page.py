from dash import dcc, html, Input, Output, callback
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import plotly.figure_factory as ff
import dash_bootstrap_components as dbc
import datetime
import plotly.graph_objects as go
import joblib
import time

# Load dataset
file_path = 'data/forest_fires_merged_with_coords.csv'
data = pd.read_csv(file_path)
data_corr = data.drop(['timestamp','region', 'area', 'Long_X', 'Lat_Y'], axis=1)

# Add month column (use timestamp if available)
if 'timestamp' in data.columns:
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['month'] = data['timestamp'].dt.month
else:
    data['month'] = np.random.randint(1, 13, size=len(data))

# Features and target
X = data[['temp', 'humidity', 'wind', 'rain', 'FFMC', 'DMC', 'DC', 'ISI', 'nino34_sst']]
y = data['fire']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models and hyperparameter grids for tuning
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Hyperparameter grids for tuning (for simplicity, let's use a small grid)
param_grids = {
    "Logistic Regression": {'C': [0.1, 1, 10]},
    "Random Forest": {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
    "XGBoost": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
    "SVM": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    "Decision Tree": {'max_depth': [None, 10, 20]}
}

# Function to train and tune models and collect metrics
def train_and_tune_models():
    results = []
    tuned_results = []

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    results_metrics = {metric: [] for metric in metrics}
    tuned_results_metrics = {metric: [] for metric in metrics}
    
    # Store confusion matrices as well
    results_cm = {}
    tuned_results_cm = {}

    for name, model in models.items():
        # Train the model with default parameters
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Collect default model metrics
        results_metrics['Accuracy'].append(acc)
        results_metrics['Precision'].append(prec)
        results_metrics['Recall'].append(rec)
        results_metrics['F1 Score'].append(f1)
        
        # Store confusion matrix for before tuning
        results_cm[name] = confusion_matrix(y_test, y_pred)
        
        # Hyperparameter tuning with GridSearchCV
        grid_search = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred_tuned = best_model.predict(X_test)
        
        tuned_acc = accuracy_score(y_test, y_pred_tuned)
        tuned_prec = precision_score(y_test, y_pred_tuned)
        tuned_rec = recall_score(y_test, y_pred_tuned)
        tuned_f1 = f1_score(y_test, y_pred_tuned)

        # Collect tuned model metrics
        tuned_results_metrics['Accuracy'].append(tuned_acc)
        tuned_results_metrics['Precision'].append(tuned_prec)
        tuned_results_metrics['Recall'].append(tuned_rec)
        tuned_results_metrics['F1 Score'].append(tuned_f1)

        # Store confusion matrix for after tuning
        tuned_results_cm[name] = confusion_matrix(y_test, y_pred_tuned)

    return results_metrics, tuned_results_metrics, results_cm, tuned_results_cm

# Get model metrics and confusion matrices before and after hyperparameter tuning
results_metrics, tuned_results_metrics, results_cm, tuned_results_cm = train_and_tune_models()

# Create DataFrame for model metrics before and after tuning
model_names = list(models.keys())
results_df = pd.DataFrame(results_metrics, index=model_names)
tuned_results_df = pd.DataFrame(tuned_results_metrics, index=model_names)

# Feature importance (Random Forest)
rf_model = models["Random Forest"]
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Define the dashboard layout with dropdowns and responsive columns
def get_home_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(html.H1('Forest Fire Prediction', 
                                           style={'textAlign': 'center', 'color': '#008000', 'font-size': '36px'})),
                    dbc.CardBody(html.P(
                        "A Comparative Study of Machine Learning Models to Predict Forest Fires",
                        style={'textAlign': 'center','font-size': '28px', 'font-weight': 'bold'}
                    )),
                ], className="mb-4 shadow-sm"),
                width=12, style={"margin-top": "20px"}
            )
        ]),

        dcc.Tabs([

            # EDA Visualizations Tab
            dcc.Tab(label='EDA Visualizations', children=[
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H4("Fire Occurrences Distribution", style={'textAlign': 'center'})),
                        dbc.CardBody(dcc.Graph(
                            figure=px.histogram(data, x='fire', color='fire', barmode='group')
                        ))
                    ], className="mb-4 shadow-sm"), width=6, style={"margin-top": "20px"}),

                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H4("Temperature Distribution Over Months", style={'textAlign': 'center'})),
                        dbc.CardBody(dcc.Graph(
                            figure=px.box(data, x='month', y='temp')
                        ))
                    ], className="mb-4 shadow-sm"), width=6, style={"margin-top": "20px"}),
                ]),

                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H4("Temperature vs Fire Class", style={'textAlign': 'center'})),
                        dbc.CardBody(dcc.Graph(
                            figure=px.histogram(data, x='temp', color='fire')
                        ))
                    ], className="mb-4 shadow-sm"), width=6),

                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H4("Features Distribution", style={'textAlign': 'center'})),
                        dbc.CardBody(dcc.Graph(
                            figure=px.box(data.melt(), x='variable', y='value')
                        ))
                    ], className="mb-4 shadow-sm"), width=6),
                ]),

                #add temperature vs humidity scatterplot chart
                #also add the multicoolinearity plot of the features exclude the region, portugal_x and portugal_y from the multicollinearity heat map
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H4("Temperature vs Humidity", style={'textAlign': 'center'})),
                        dbc.CardBody(dcc.Graph(
                            figure=px.scatter(data, x='temp', y='humidity', color='fire', size='temp', hover_name='timestamp')
                        ))
                    ], className="mb-4 shadow-sm"), width=6, style={"margin-top": "20px"}),

                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H4("Multicollinearity Heat Map", style={'textAlign': 'center'})),
                        dbc.CardBody(dcc.Graph(
                            figure=px.imshow(data_corr.corr(), text_auto=True)
                        ))
                    ], className="mb-4 shadow-sm"), width=6, style={"margin-top": "20px"}),
                ]),

                #add a bubble geographical map of the fires in algeria
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H4("Fires Geographical Map of Algeria and Portugal", style={'textAlign': 'center'})),
                        dbc.CardBody(dcc.Graph(
                            figure=px.scatter_mapbox(data, lat="Lat_Y", lon="Long_X", color="fire", size="temp", hover_name="timestamp", mapbox_style="open-street-map", zoom=5, height=600, hover_data=["region", "temp", "humidity", 'wind', 'rain', 'FFMC', 'DMC'])
                        ))
                    ], className="mb-4 shadow-sm"), width=12, style={"margin-top": "20px"}),
                ]),

                #use the model to create a weekly prediction chart of the probable fire events after choosing a date 
                # dbc.Row([
                #     dbc.Col(dbc.Card([
                #         dbc.CardHeader(html.H4("Weekly Fire Prediction", style={'textAlign': 'center'})),
                #         dbc.CardBody(children=[
                #             html.Label("Select a start date:"),
                #             dcc.DatePickerSingle(
                #                 id="start-date",
                #                 min_date_allowed=data['timestamp'].min(),
                #                 max_date_allowed=data['timestamp'].max(),
                #                 initial_visible_month=data['timestamp'].max(),
                #                 date=data['timestamp'].max(),
                #             ),
                #             dcc.Graph(id="prediction-chart"),
                #         ])
                #     ], className="mb-4 shadow-sm"), width=12, style={"margin-top": "20px"}),
                # ]),
            ]),

            # Model Performance Metrics Tab
            dcc.Tab(label='Model Performance Metrics', children=[
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H4("Accuracy Comparison between different Machine Learing Models", style={'textAlign': 'center'})),
                        dbc.CardBody(dcc.Graph(
                            figure=px.bar(
                                pd.DataFrame({
                                    'Model': model_names,
                                    'Before Tuning': results_df['Accuracy'],
                                    'After Tuning': tuned_results_df['Accuracy']
                                }),
                                x='Model', 
                                y=['Before Tuning', 'After Tuning'],
                                barmode='group',
                            )
                        ))
                    ], className="mb-4 shadow-sm"), width=12, style={"margin-top": "20px"}),
                ]),

                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H4("Precision Comparison between different Machine Learning Models", style={'textAlign': 'center'})),
                        dbc.CardBody(dcc.Graph(
                            figure=px.bar(
                                pd.DataFrame({
                                    'Model': model_names,
                                    'Before Tuning': results_df['Precision'],
                                    'After Tuning': tuned_results_df['Precision']
                                }),
                                x='Model', 
                                y=['Before Tuning', 'After Tuning'],
                                barmode='group',
                            )
                        ))
                    ], className="mb-4 shadow-sm"), width=12),
                ]),

                # Add Recall and F1 Score Rows similarly
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H4("Recall Comparison between different Machine Learning Models", style={'textAlign': 'center'})),
                        dbc.CardBody(dcc.Graph(
                            figure=px.bar(
                                pd.DataFrame({
                                    'Model': model_names,
                                    'Before Tuning': results_df['Recall'],
                                    'After Tuning': tuned_results_df['Recall']
                                }),
                                x='Model', 
                                y=['Before Tuning', 'After Tuning'],
                                barmode='group',
                            )
                        ))
                    ], className="mb-4 shadow-sm"), width=12),
                ]),

                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H4("F1 Score Comparison between different Machine Learning Models", style={'textAlign': 'center'})),
                        dbc.CardBody(dcc.Graph(
                            figure=px.bar(
                                pd.DataFrame({
                                    'Model': model_names,
                                    'Before Tuning': results_df['F1 Score'],
                                    'After Tuning': tuned_results_df['F1 Score']
                                }),
                                x='Model', 
                                y=['Before Tuning', 'After Tuning'],
                                barmode='group',
                            )
                        ))
                    ], className="mb-4 shadow-sm"), width=12),
                ])
            ]),

            # Confusion Matrix Tab
            dcc.Tab(label='Confusion Matrix', children=[
                dbc.Card([
                    dbc.CardHeader(html.H4("Select Model to View Confusion Matrix", style={'textAlign': 'center'})),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='model-dropdown',
                            options=[{'label': model, 'value': model} for model in model_names],
                            value=model_names[0],  # Default to the first model
                            style={'width': '50%', 'margin': 'auto'}
                        ),
                        html.Div(id='confusion-matrix-container', style={'textAlign': 'center', 'margin-top': '30px'}),
                    ])
                ], className="mb-4 shadow-sm")
            ]),

            # Feature Importance Tab
            dcc.Tab(label='Feature Importance', children=[
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H4("Feature Importance", style={'textAlign': 'center'})),
                        dbc.CardBody(dcc.Graph(
                            figure=px.bar(
                                feature_importance_df, x='Feature', y='Importance',
                                title='Feature Importance'
                            )
                        ))
                    ], className="mb-4 shadow-sm"), width=12),
                ]),
            ]),
        ])
    ], fluid=True)

# Callback to update confusion matrix based on dropdown selection
@callback(
    Output('confusion-matrix-container', 'children'),
    Input('model-dropdown', 'value')
)
def update_confusion_matrix(selected_model):
    # Find the corresponding confusion matrix (before and after tuning)
    cm_before = results_cm[selected_model]
    cm_after = tuned_results_cm[selected_model]

    # Generate the confusion matrix plots
    cm_figure = ff.create_annotated_heatmap(
        z=cm_before, 
        x=['No Fire', 'Fire'], 
        y=['No Fire', 'Fire'],
        annotation_text=cm_before.astype(str), 
        colorscale='Viridis'
    )
    
    cm_figure_after = ff.create_annotated_heatmap(
        z=cm_after, 
        x=['No Fire', 'Fire'], 
        y=['No Fire', 'Fire'],
        annotation_text=cm_after.astype(str), 
        colorscale='Viridis'
    )

    return html.Div([
        # html.H4(f"Confusion Matrix - Before Tuning: {selected_model}", style={'textAlign': 'center'}),
        # dcc.Graph(figure=cm_figure),
        html.H4(f"Confusion Matrix", style={'textAlign': 'center'}),
        dcc.Graph(figure=cm_figure_after)
    ])

# Callback for predictions
# @callback(
#     Output("prediction-chart", "figure"),
#     Input("start-date", "date")
# )
# def update_chart(start_date):
#     data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
#     data['fire'] = data['fire'].astype(int)

#     # Load pretrained model
#     model_path = "best_model.pkl"  # Replace with your model path
#     model = joblib.load(model_path)

#     # Parse the selected start date
#     start_date = pd.to_datetime(start_date)
    
#     # Generate predictions (example: naive method using historical averages)
#     daily_avg_fire = data.groupby(data['timestamp'].dt.dayofweek)['fire'].mean()
    
#     # Predict for the next 7 days
#     prediction_dates = [start_date + datetime.timedelta(days=i) for i in range(7)]
#     #predictions = [daily_avg_fire[date.weekday()] for date in prediction_dates]
#     # feature_data = pd.DataFrame({
#     #     "day_of_week": [date.weekday() for date in prediction_dates],
#     #     "month": [date.month for date in prediction_dates],
#     #     # Add other relevant features as needed
#     # })
#     feature_data = pd.DataFrame({
        
#     })

#     # Predict fire probabilities using the pretrained model
#     predictions = model.predict(feature_data)

#     # Create the prediction chart
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=prediction_dates,
#         y=predictions,
#         mode='lines+markers',
#         name="Predicted Fires"
#     ))
#     fig.update_layout(
#         title="Weekly Fire Predictions",
#         xaxis_title="Date",
#         yaxis_title="Predicted Fire Events",
#         template="plotly_white"
#     )
#     return fig