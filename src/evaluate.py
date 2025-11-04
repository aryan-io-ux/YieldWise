# src/evaluate.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from model_utils import load_pipeline

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'grain_predictor_pipeline.joblib'
TEST_DATA_PATH = PROJECT_ROOT / 'data' / 'test.csv'
TARGET_COLUMN = 'remaining_safe_storage_days'

def evaluate_model():
    """
    Loads the test data and the trained model, makes predictions,
    and generates an enhanced 'Predictions vs. Actuals' scatter plot.
    """
    print("Starting model evaluation...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    X_test = df_test.drop(TARGET_COLUMN, axis=1)
    y_test_actual = df_test[TARGET_COLUMN]

    pipeline = load_pipeline(MODEL_PATH)
    y_test_predicted = pipeline.predict(X_test)
    print("Predictions generated successfully.")

    # --- Create the Plot ---
    results_df = pd.DataFrame({
        'Actual Safe Days': y_test_actual,
        'Predicted Safe Days': y_test_predicted
    })

    # Create the scatter plot WITH A TRENDLINE
    fig = px.scatter(
        results_df,
        x='Actual Safe Days',
        y='Predicted Safe Days',
        title='Model Evaluation: Predictions vs. Actual Values',
        labels={'Actual Safe Days': 'Actual Values (Days)', 'Predicted Safe Days': 'Predicted Values (Days)'},
        trendline="ols",  # This adds a linear trendline to the scatter plot
        trendline_color_override="green" # Make the trendline green
    )

    # Add the 45-degree "Perfect Prediction" line for reference
    min_val = min(y_test_actual.min(), y_test_predicted.min())
    max_val = max(y_test_actual.max(), y_test_predicted.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        width=800, height=800,
        xaxis_title='Actual Safe Days', yaxis_title='Predicted Safe Days',
        legend=dict(x=0.01, y=0.99)
    )

    print("Displaying enhanced evaluation plot...")
    fig.show()

if __name__ == "__main__":
    evaluate_model()