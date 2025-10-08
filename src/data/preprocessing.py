"""
Data loading and preprocessing utilities
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
import pickle

from ..config.settings import TITANIC_CSV, FEATURES_TO_DROP


def load_raw_data(file_path: Path = TITANIC_CSV) -> pd.DataFrame:
    """Load raw Titanic dataset"""
    try:
        data = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        raise
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


def drop_unnecessary_columns(data: pd.DataFrame, columns_to_drop: List[str] = FEATURES_TO_DROP) -> pd.DataFrame:
    """Drop unnecessary columns from the dataset"""
    data_cleaned = data.drop(columns_to_drop, axis=1)
    print(f"✅ Dropped columns: {columns_to_drop}")
    print(f"Remaining columns: {list(data_cleaned.columns)}")
    return data_cleaned


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset"""
    print(f"Missing values before cleaning:\n{data.isnull().sum()}")
    data_cleaned = data.dropna()
    print(f"✅ Dropped rows with missing values. New shape: {data_cleaned.shape}")
    return data_cleaned


def encode_categorical_variables(data: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables using one-hot encoding"""
    data_encoded = pd.get_dummies(data, drop_first=True)
    print(f"✅ Categorical variables encoded. New shape: {data_encoded.shape}")
    print(f"Final columns: {list(data_encoded.columns)}")
    return data_encoded


def split_features_target(data: pd.DataFrame, target_column: str = "Survived") -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and target variable"""
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    print(f"✅ Features and target split. Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y


def scale_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✅ Features scaled. Shape: {X_scaled.shape}")
    return X_scaled, scaler


def save_preprocessing_objects(scaler: StandardScaler, feature_names: List[str], 
                             significant_features: List[str], models_path: Path):
    """Save preprocessing objects for future use"""
    scaler_file = models_path / "scaler.pkl"
    feature_names_file = models_path / "feature_names.pkl"
    significant_features_file = models_path / "significant_features.pkl"
    
    pickle.dump(scaler, open(scaler_file, 'wb'))
    pickle.dump(feature_names, open(feature_names_file, 'wb'))
    pickle.dump(significant_features, open(significant_features_file, 'wb'))
    
    print("✅ Preprocessing objects saved:")
    print(f"   - Scaler: {scaler_file}")
    print(f"   - Feature names: {feature_names_file}")
    print(f"   - Significant features: {significant_features_file}")