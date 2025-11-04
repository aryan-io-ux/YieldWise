# src/data_preprocessing.py

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_preprocessor(numerical_features, categorical_features):
    """
    Creates a scikit-learn preprocessing pipeline.

    This pipeline scales numerical features and one-hot encodes
    categorical features.
    """
    
    # Create preprocessing steps for numerical data (scaling)
    numerical_transformer = StandardScaler()

    # Create preprocessing steps for categorical data (one-hot encoding)
    # handle_unknown='ignore' prevents errors if a new category appears in prediction data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine preprocessing steps into a single object
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor