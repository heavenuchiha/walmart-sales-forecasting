import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from joblib import load

# Load your data
data_path = "C:/Users/heave/Videos/data/Sales Walmart/walmart-recruiting-store-sales-forecasting/merged_cleaned.csv"  # Update if needed
model_path = "C:/Users/heave/Videos/data/Sales Walmart/walmart-recruiting-store-sales-forecasting/xgb_sales_model.joblib"

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    df = pd.DataFrame()

# Load the model
try:
    model = load(model_path)
except Exception as e:
    print("Model could not be loaded:", e)
    model = None

# App initialization
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Walmart Sales Forecasting"

app.layout = dbc.Container([
    html.H1("📊 Walmart Sales Forecasting Dashboard", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Store"),
            dcc.Dropdown(id="store", options=[{"label": str(s), "value": s} for s in sorted(df['Store'].unique())], value=1),
            
            html.Label("Department"),
            dcc.Dropdown(id="dept", options=[{"label": str(d), "value": d} for d in sorted(df['Dept'].unique())], value=1),
            
            html.Label("Store Size"),
            dcc.Input(id="size", type="number", value=150000),
            
            html.Label("Is Holiday"),
            dcc.Checklist(id="isholiday", options=[{"label": "Yes", "value": 1}], value=[]),
            
            html.Label("Temperature"),
            dcc.Slider(id="temp", min=0, max=120, step=1, value=70, tooltip={"placement": "bottom"}),
            
            html.Label("Fuel Price"),
            dcc.Slider(id="fuel", min=2.0, max=4.5, step=0.1, value=3.0, tooltip={"placement": "bottom"}),
            
            html.Label("CPI"),
            dcc.Slider(id="cpi", min=100.0, max=250.0, step=1.0, value=150.0),
            
            html.Label("Unemployment"),
            dcc.Slider(id="unemp", min=4.0, max=12.0, step=0.1, value=7.0),
            
            html.Label("Year"),
            dcc.Input(id="year", type="number", value=2012),
            
            html.Label("Month"),
            dcc.Slider(id="month", min=1, max=12, step=1, value=6),
            
            html.Label("Week"),
            dcc.Slider(id="week", min=1, max=52, step=1, value=26),
            
            html.Label("Store Type"),
            dcc.Checklist(id="type", options=[
                {"label": "Type B", "value": "B"},
                {"label": "Type C", "value": "C"}
            ], value=[]),

            html.Br(),
            dbc.Button("Predict Sales", id="predict_btn", color="primary", className="w-100"),
            html.Div(id="prediction_output", className="mt-3 text-success fw-bold")
        ], md=4),

        dbc.Col([
            html.H5("📈 CPI vs Weekly Sales", className="mb-2"),
            dcc.Graph(id="cpi_chart")
        ], md=8)
    ])
], fluid=True)

# Callback for prediction
@app.callback(
    Output("prediction_output", "children"),
    Input("predict_btn", "n_clicks"),
    State("Store", "value"),
    State("dept", "value"),
    State("size", "value"),
    State("isholiday", "value"),
    State("temp", "value"),
    State("fuel", "value"),
    State("cpi", "value"),
    State("unemp", "value"),
    State("year", "value"),
    State("month", "value"),
    State("week", "value"),
    State("type", "value")
)
def predict_sales(n_clicks, store, dept, size, isholiday, temp, fuel, cpi, unemp, year, month, week, store_type):
    if n_clicks is None or model is None:
        return ""

    type_b = 1 if "B" in store_type else 0
    type_c = 1 if "C" in store_type else 0

    input_data = pd.DataFrame([{
        'Store': store,
        'Dept': dept,
        'Size': size,
        'IsHoliday': 1 if isholiday else 0,
        'Temperature': temp,
        'Fuel_Price': fuel,
        'CPI': cpi,
        'Unemployment': unemp,
        'Year': year,
        'Month': month,
        'Week': week,
        'Type_B': type_b,
        'Type_C': type_c
    }])

    try:
        prediction = model.predict(input_data)[0]
        return f"📊 Predicted Weekly Sales: ${prediction:,.2f}"
    except Exception as e:
        return f"⚠️ Error making prediction: {e}"

# Callback for CPI vs Weekly Sales chart
@app.callback(
    Output("cpi_chart", "figure"),
    Input("store", "value")
)
def update_cpi_chart(store):
    if 'CPI' not in df.columns or 'Weekly_Sales' not in df.columns:
        return px.scatter(title="Missing 'CPI' or 'Weekly_Sales' column")

    filtered = df[df["Store"] == store]
    fig = px.scatter(filtered, x="CPI", y="Weekly_Sales", title=f"CPI vs Sales (Store {store})")
    return fig

if __name__ == "__main__":
    app.run(debug=True)
