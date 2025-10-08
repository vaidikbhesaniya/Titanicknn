"""
Model training utilities
"""
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
from typing import Tuple

from ..config.settings import MODEL_PARAMS, MODEL_FILE


def split_train_test(X: np.ndarray, y: np.ndarray, 
                    test_size: float = MODEL_PARAMS["test_size"], 
                    random_state: int = MODEL_PARAMS["random_state"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets
    
    Args:
        X: Feature array
        y: Target array
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✅ Data split completed:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set: {X_test.shape[0]} samples")
    print(f"   Test size: {test_size * 100}%")
    
    return X_train, X_test, y_train, y_test


def train_knn_model(X_train: np.ndarray, y_train: np.ndarray, 
                   n_neighbors: int = MODEL_PARAMS["n_neighbors"]) -> KNeighborsClassifier:
    """
    Train a K-Nearest Neighbors classifier
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_neighbors: Number of neighbors for KNN
    
    Returns:
        Trained KNN model
    """
    print(f"\n=== Training KNN Model (k={n_neighbors}) ===")
    
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    
    print(f"✅ KNN model trained successfully")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   K neighbors: {n_neighbors}")
    
    return model


def save_model(model: KNeighborsClassifier, model_path: Path = MODEL_FILE) -> None:
    """
    Save trained model to disk
    
    Args:
        model: Trained model
        model_path: Path to save the model
    """
    # Ensure the models directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    pickle.dump(model, open(model_path, 'wb'))
    print(f"✅ Model saved to: {model_path}")


def load_model(model_path: Path = MODEL_FILE) -> KNeighborsClassifier:
    """
    Load trained model from disk
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Loaded model
    """
    try:
        model = pickle.load(open(model_path, 'rb'))
        print(f"✅ Model loaded from: {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        raise
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise