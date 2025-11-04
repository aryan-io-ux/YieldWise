# src/train.py

import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Import our custom modules
from data_preprocessing import create_preprocessor
from model_utils import define_model, save_pipeline

# --- Configuration ---
TRAIN_DATA_PATH = '../data/train.csv'
TEST_DATA_PATH = '../data/test.csv'
MODEL_SAVE_PATH = '../models/grain_predictor_pipeline.joblib'
METRICS_SAVE_PATH = '../output/metrics.json'
TARGET_COLUMN = 'remaining_safe_storage_days'

# --- Main Training Function ---
def run_training():
    """Orchestrates the model training process."""
    print("Starting model training process...")

    # Load data
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    # Separate features (X) and target (y)
    X_train = df_train.drop(TARGET_COLUMN, axis=1)
    y_train = df_train[TARGET_COLUMN]
    X_test = df_test.drop(TARGET_COLUMN, axis=1)
    y_test = df_test[TARGET_COLUMN]

    # Identify feature types
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    # Create the full pipeline (preprocessor + model)
    preprocessor = create_preprocessor(numerical_features, categorical_features)
    model = define_model()
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # Train the pipeline
    print("Training the model...")
    pipeline.fit(X_train, y_train)

    # Evaluate the model on the test set
    print("Evaluating the model...")
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  Mean Absolute Error (MAE): {mae:.2f} days")
    print(f"  R-squared (R²): {r2:.2f}")

    # Save metrics
    metrics = {'mean_absolute_error': mae, 'r2_score': r2}
    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {METRICS_SAVE_PATH}")

    # Save the trained pipeline
    save_pipeline(pipeline, MODEL_SAVE_PATH)

if __name__ == "__main__":
    run_training()