import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path
from src.model_utils import load_pipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="YieldWise", page_icon="🌱", layout="wide")


# --- PATHS & DATA LOADING ---
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'grain_predictor_pipeline.joblib'
METRICS_PATH = PROJECT_ROOT / 'output' / 'metrics.json'
TEST_DATA_PATH = PROJECT_ROOT / 'data' / 'test.csv'
TARGET_COLUMN = 'remaining_safe_storage_days'


# --- MODEL AND DATA LOADING (CACHED) ---
@st.cache_resource
def load_model_pipeline():
    return load_pipeline(MODEL_PATH)

@st.cache_data
def load_data():
    try:
        metrics = json.load(open(METRICS_PATH))
    except FileNotFoundError:
        metrics = {} # Handle case where metrics file doesn't exist yet
    df_test = pd.read_csv(TEST_DATA_PATH)
    return metrics, df_test

pipeline = load_model_pipeline()
metrics, df_test = load_data()


# --- STYLING ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #82a89d; }
    .main { background-color: #F5F5F5; padding: 2rem; }
    .stButton>button { width: 100%; border-radius: 10px; background-color: #4CAF50; color: white; border: none; }
    </style>
    """, unsafe_allow_html=True)


# --- HEADER ---
st.title("🌱 YieldWise: Storage Optimizer")


# --- TABS FOR NAVIGATION ---
tab1, tab2 = st.tabs(["🔮 Live Prediction & Analysis", "📊 Model Performance"])


# --- LIVE PREDICTION & ANALYSIS TAB ---
with tab1:
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.header("Silo Conditions")
        user_inputs = {}
        user_inputs['grain_type'] = st.selectbox("Grain Type", options=['Wheat', 'Corn', 'Rice', 'Barley'], index=1)
        user_inputs['avg_internal_temp_c'] = st.slider("Average Internal Temp (°C)", 5.0, 40.0, 25.0, 0.5)
        user_inputs['initial_moisture_content'] = st.slider("Initial Moisture Content (%)", 10.0, 20.0, 14.5, 0.1)
        user_inputs['days_since_fill'] = st.slider("Days Since Fill", 1, 365, 30)
        aeration_status_str = st.radio("Aeration System", ['OFF', 'ON'], horizontal=True)
        user_inputs['aeration_status'] = 1 if aeration_status_str == 'ON' else 0

    with col2:
        st.header("Forecast & What-If Analysis")
        # --- PREDICTION LOGIC ---
        input_df = pd.DataFrame([user_inputs])
        predicted_rssd = pipeline.predict(input_df)[0]

        # --- GAUGE CHART ---
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=predicted_rssd,
            title={'text': "Predicted Safe Storage Days (RSSD)", 'font': {'size': 20}},
            gauge={'axis': {'range': [None, 250]},
                   'steps': [{'range': [0, 30], 'color': '#EA4335'},
                             {'range': [30, 90], 'color': '#FBBC04'},
                             {'range': [90, 250], 'color': '#34A853'}]}))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # --- LINE CHART FOR WHAT-IF ANALYSIS ---
        st.markdown("---")
        st.subheader("Impact of Temperature on Storage Safety")
        
        # Create a range of temperatures to simulate
        temp_range = np.arange(5.0, 40.5, 0.5)
        simulation_results = []

        # Run simulation
        for temp in temp_range:
            sim_inputs = user_inputs.copy()
            sim_inputs['avg_internal_temp_c'] = temp
            sim_df = pd.DataFrame([sim_inputs])
            prediction = pipeline.predict(sim_df)[0]
            simulation_results.append(prediction)

        # Create the line chart
        fig_line = px.line(x=temp_range, y=simulation_results, 
                           labels={'x': 'Average Internal Temperature (°C)', 'y': 'Predicted Safe Days'},
                           title="How RSSD Changes with Temperature")
        
        # Add a vertical line to show the user's current selection
        fig_line.add_vline(x=user_inputs['avg_internal_temp_c'], line_width=3, line_dash="dash", line_color="red", 
                          annotation_text="Your Selection", annotation_position="top right")
        
        st.plotly_chart(fig_line, use_container_width=True)

# --- MODEL PERFORMANCE TAB ---
with tab2:
    st.header("Understanding the Model's Performance")
    st.write("These metrics are calculated from the unseen test dataset (`test.csv`).")

    if metrics:
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("R-squared (R²)", f"{metrics.get('r2_score', 0):.2f}", "Measures model fit.")
        mcol2.metric("Mean Absolute Error (MAE)", f"{metrics.get('mean_absolute_error', 0):.2f} Days", "Average prediction error.")

    else:
        st.warning("Metrics file not found. Please run `src/train.py` to generate it.")
    
    st.markdown("---")
    
    st.subheader("Predictions vs. Actual Values Scatter Plot")
    X_test = df_test.drop(TARGET_COLUMN, axis=1)
    y_test_actual = df_test[TARGET_COLUMN]
    y_test_predicted = pipeline.predict(X_test)
    results_df = pd.DataFrame({'Actual': y_test_actual, 'Predicted': y_test_predicted})
    
    fig = px.scatter(results_df, x='Actual', y='Predicted', 
                     title="Visualizing Prediction Accuracy",
                     labels={'Actual': 'Actual Safe Days', 'Predicted': 'Predicted Safe Days'},
                     marginal_x="histogram", marginal_y="histogram")
    fig.add_shape(type="line", x0=0, y0=0, x1=250, y1=250, line=dict(color="Red", width=2, dash="dash"))
    st.plotly_chart(fig, use_container_width=True)