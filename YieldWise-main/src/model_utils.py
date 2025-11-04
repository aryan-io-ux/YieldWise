# src/model_utils.py

import joblib
from sklearn.ensemble import GradientBoostingRegressor

def define_model():
    """
    Defines the machine learning model with specific hyperparameters.
    We use Gradient Boosting, a powerful algorithm for tabular data.
    """
    model = GradientBoostingRegressor(
        n_estimators=100,      # Number of boosting stages
        learning_rate=0.1,     # How much each tree contributes
        max_depth=5,           # Maximum depth of individual trees
        random_state=42        # Ensures reproducibility
    )
    return model

def save_pipeline(pipeline, file_path):
    """Saves the entire pipeline (preprocessor + model) to a file."""
    print(f"Saving pipeline to {file_path}...")
    joblib.dump(pipeline, file_path)
    print("Pipeline saved successfully.")

def load_pipeline(file_path):
    """Loads a pipeline from a file."""
    print(f"Loading pipeline from {file_path}...")
    pipeline = joblib.load(file_path)
    print("Pipeline loaded successfully.")
    return pipeline