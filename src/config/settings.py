"""
Configuration settings for the Titanic ML project
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL_PATH = PROJECT_ROOT / "data" / "external"

# Model paths
MODELS_PATH = PROJECT_ROOT / "models"

# Raw data file
TITANIC_CSV = DATA_RAW_PATH / "Titanic.csv"

# Model files
MODEL_FILE = MODELS_PATH / "titanic_knn_model.pkl"
SCALER_FILE = MODELS_PATH / "scaler.pkl"
FEATURE_NAMES_FILE = MODELS_PATH / "feature_names.pkl"
SIGNIFICANT_FEATURES_FILE = MODELS_PATH / "significant_features.pkl"

# Model parameters
MODEL_PARAMS = {
    "n_neighbors": 5,
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "significance_threshold": 0.05
}

# Features to drop
FEATURES_TO_DROP = ['Name', 'Ticket', 'Fare', 'Cabin']