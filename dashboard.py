import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from joblib import load
from datetime import datetime, timedelta

st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")

# Load cleaned data
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/heave/Videos/data/Sales Walmart/walmart-recruiting-store-sales-forecasting/merged_cleaned.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

data = load_data()

# Load model
model = load("C:/Users/heave/Videos/data/Sales Walmart/walmart-recruiting-store-sales-forecasting/xgb_sales_model.joblib")

# Metrics
MAE = 12094.57  # Your calculated MAE
RMSE = 18765.31  # Your calculated RMSE

# Load plots
feature_img = Image.open("C:/Users/heave/Videos/data/Sales Walmart/walmart-recruiting-store-sales-forecasting/plots/feature_importance.png")
store_img = Image.open("C:/Users/heave/Videos/data/Sales Walmart/walmart-recruiting-store-sales-forecasting/plots/store_sales_comparison.png")
residual_img = Image.open("C:/Users/heave/Videos/data/Sales Walmart/walmart-recruiting-store-sales-forecasting/plots/residual_distribution.png")

# App Title
st.title("📊 Retail Sales Forecasting Dashboard")
st.markdown("Built using XGBoost and Streamlit.")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Model Evaluation", "🎯 Predict Sales", "⏳ Forecast Over Time"])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.header("📌 Key Performance Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error", f"${MAE:,.2f}")
    col2.metric("Root Mean Squared Error", f"${RMSE:,.2f}")

    st.subheader("🔧 Feature Importance")
    st.image(feature_img, use_container_width=True)  # Updated parameter

    st.subheader("🏪 Store-Level Sales: Actual vs Predicted")
    st.image(store_img, use_container_width=True)  # Updated parameter

    st.subheader("📉 Prediction Error Distribution")
    st.image(residual_img, use_container_width=True)  # Updated parameter

# --- TAB 2: MODEL EVALUATION ---
with tab2:
    st.header("📊 In-Depth Evaluation")

    fluctuation = data.groupby('Store')['Weekly_Sales'].std().sort_values(ascending=False).reset_index()
    st.write("Top Stores with Most Fluctuating Weekly Sales:")
    st.dataframe(fluctuation.head(10))

    st.subheader("🧾 CPI vs Weekly Sales")
    st.scatter_chart(data[['CPI', 'Weekly_Sales']])

    st.subheader("🎉 Holiday vs Non-Holiday Sales")
    holiday_sales = data.groupby('IsHoliday')['Weekly_Sales'].mean().reset_index()
    st.bar_chart(holiday_sales.set_index('IsHoliday'))

# --- TAB 3: MAKE A PREDICTION ---
with tab3:
    st.header("📈 Forecast Weekly Sales")
    
    # Prediction form
    store = st.selectbox("Store", sorted(data['Store'].unique()))
    dept = st.selectbox("Department", sorted(data['Dept'].unique()))
    size = st.number_input("Store Size", value=150000)
    isholiday = st.checkbox("Is Holiday", value=False)
    temp = st.slider("Temperature (°F)", 0.0, 120.0, 70.0)
    fuel = st.slider("Fuel Price ($)", 2.0, 4.5, 3.0)
    cpi = st.slider("CPI", 100.0, 250.0, 150.0)
    unemp = st.slider("Unemployment (%)", 4.0, 12.0, 7.0)
    year = st.number_input("Year", value=2012)
    month = st.slider("Month", 1, 12, 6)
    week = st.slider("Week", 1, 52, 26)
    store_type = st.selectbox("Store Type", ["A", "B", "C"])  # User selects type
    
    # Convert type to one-hot encoded columns
    type_a = 1 if store_type == "A" else 0
    type_b = 1 if store_type == "B" else 0
    type_c = 1 if store_type == "C" else 0
    
    md1 = st.number_input("MarkDown1", value=0.0)
    md2 = st.number_input("MarkDown2", value=0.0)
    md3 = st.number_input("MarkDown3", value=0.0)
    md4 = st.number_input("MarkDown4", value=0.0)
    md5 = st.number_input("MarkDown5", value=0.0)

    if st.button("Predict Sales"):
        input_df = pd.DataFrame({
            'Store': [store],
            'Dept': [dept],
            'IsHoliday': [isholiday],
            'Temperature': [temp],
            'Fuel_Price': [fuel],
            'MarkDown1': [md1],
            'MarkDown2': [md2],
            'MarkDown3': [md3],
            'MarkDown4': [md4],
            'MarkDown5': [md5],
            'CPI': [cpi],
            'Unemployment': [unemp],
            'Size': [size],
            'Year': [year],
            'Month': [month],
            'Week': [week],
            'Type_A': [type_a],  # One-hot encoded
            'Type_B': [type_b],  # One-hot encoded
            'Type_C': [type_c]   # One-hot encoded
        })

        prediction = model.predict(input_df)[0]
        st.success(f"🤑 Predicted Weekly Sales: **${prediction:,.2f}**")
        
        # Add export button for single prediction
        export_df = input_df.copy()
        export_df['Predicted_Sales'] = prediction
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Prediction",
            data=csv,
            file_name=f"sales_prediction_store_{store}_dept_{dept}.csv",
            mime="text/csv"
        )

# --- TAB 4: FORECAST OVER TIME ---
with tab4:
    st.header("⏳ Time Series Forecast")
    
    col1, col2 = st.columns(2)
    selected_store = col1.selectbox("Select Store", sorted(data['Store'].unique()), key='store_forecast')
    selected_dept = col2.selectbox("Select Department", sorted(data['Dept'].unique()), key='dept_forecast')
    
    weeks_to_forecast = st.slider("Number of Weeks to Forecast", 1, 52, 12)
    
    if st.button("Generate Forecast"):
        # Get the most recent data for the selected store and department
        latest_data = data[(data['Store'] == selected_store) & 
                          (data['Dept'] == selected_dept)].sort_values('Date').tail(1)
        
        if len(latest_data) == 0:
            st.warning("No historical data found for this store and department combination.")
        else:
            # Prepare the base data for forecasting
            base_row = latest_data.iloc[0]
            forecasts = []
            dates = []
            
            # Get store size (assuming it's constant for each store)
            store_size = base_row['Size']
            
            # Get store type (convert from 'Type' column to one-hot encoded)
            store_type = base_row['Type']
            type_a = 1 if store_type == "A" else 0
            type_b = 1 if store_type == "B" else 0
            type_c = 1 if store_type == "C" else 0
            
            # Generate forecasts for each week
            for i in range(1, weeks_to_forecast + 1):
                # Calculate next date (assuming weekly intervals)
                next_date = base_row['Date'] + timedelta(weeks=i)
                
                # Prepare input data
                input_data = {
                    'Store': selected_store,
                    'Dept': selected_dept,
                    'IsHoliday': False,
                    'Temperature': base_row['Temperature'],
                    'Fuel_Price': base_row['Fuel_Price'],
                    'MarkDown1': base_row['MarkDown1'],
                    'MarkDown2': base_row['MarkDown2'],
                    'MarkDown3': base_row['MarkDown3'],
                    'MarkDown4': base_row['MarkDown4'],
                    'MarkDown5': base_row['MarkDown5'],
                    'CPI': base_row['CPI'],
                    'Unemployment': base_row['Unemployment'],
                    'Size': store_size,
                    'Year': next_date.year,
                    'Month': next_date.month,
                    'Week': next_date.isocalendar()[1],
                    'Type_A': type_a,  # One-hot encoded
                    'Type_B': type_b,  # One-hot encoded
                    'Type_C': type_c   # One-hot encoded
                }
                
                # Create DataFrame and predict
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                
                forecasts.append(prediction)
                dates.append(next_date)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'Date': dates,
                'Store': selected_store,
                'Dept': selected_dept,
                'Predicted_Sales': forecasts
            })
            
            # Display forecast
            st.subheader(f"Forecast for Store {selected_store}, Department {selected_dept}")
            st.line_chart(forecast_df.set_index('Date')['Predicted_Sales'])
            
            # Show data table
            st.dataframe(forecast_df)
            
            # Export forecast data
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Forecast Data",
                data=csv,
                file_name=f"sales_forecast_store_{selected_store}_dept_{selected_dept}.csv",
                mime="text/csv"
            )