# src/predict.py

import pandas as pd
from model_utils import load_pipeline

# --- Configuration ---
MODEL_PATH = '../models/grain_predictor_pipeline.joblib'

def predict(input_data):
    """Loads the pipeline and makes a prediction."""
    
    # Load the trained pipeline
    pipeline = load_pipeline(MODEL_PATH)
    
    # Create a DataFrame from the input data
    # The column names MUST match the training data
    input_df = pd.DataFrame([input_data])
    
    # Make a prediction
    prediction = pipeline.predict(input_df)
    
    return prediction[0]

if __name__ == "__main__":
    # Example of new data for a silo
    sample_data = {
        'grain_type': 'Corn',
        'initial_moisture_content': 15.5,
        'avg_internal_temp_c': 28.0,
        'days_since_fill': 30,
        'aeration_status': 0 # Aeration is OFF
    }
    
    predicted_rssd = predict(sample_data)
    
    print("\n--- Grain Storage Prediction ---")
    print(f"Input Data: {sample_data}")
    print(f"Predicted Remaining Safe Storage Days (RSSD): {predicted_rssd:.0f} days")
    print("----------------------------")