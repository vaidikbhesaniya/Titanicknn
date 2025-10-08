"""
Model evaluation utilities
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, Tuple

from ..config.settings import MODEL_PARAMS


def evaluate_train_test_split(model: KNeighborsClassifier, X_test: np.ndarray, 
                            y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance on train-test split
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True test labels
        y_pred: Predicted test labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    print("\n=== Train-Test Split Evaluation ===")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    return metrics


def evaluate_cross_validation(model: KNeighborsClassifier, X: np.ndarray, y: np.ndarray, 
                            cv_folds: int = MODEL_PARAMS["cv_folds"]) -> Dict[str, float]:
    """
    Evaluate model performance using cross-validation
    
    Args:
        model: Trained model
        X: All features
        y: All targets
        cv_folds: Number of cross-validation folds
    
    Returns:
        Dictionary with cross-validation metrics
    """
    print(f"\n=== {cv_folds}-Fold Cross-Validation Evaluation ===")
    
    # Accuracy scores for each fold
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
    print(f"CV Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
    
    # Cross-validation predictions for precision and recall
    y_cv_pred = cross_val_predict(model, X, y, cv=cv_folds)
    
    cv_metrics = {
        'cv_accuracy_mean': np.mean(cv_scores),
        'cv_accuracy_std': np.std(cv_scores),
        'cv_precision': precision_score(y, y_cv_pred),
        'cv_recall': recall_score(y, y_cv_pred)
    }
    
    print("\n=== Cross-Validation Metrics ===")
    print(f"Accuracy: {cv_metrics['cv_accuracy_mean']:.4f} (±{cv_metrics['cv_accuracy_std']:.4f})")
    print(f"Precision: {cv_metrics['cv_precision']:.4f}")
    print(f"Recall: {cv_metrics['cv_recall']:.4f}")
    
    return cv_metrics


def compare_evaluation_methods(train_test_accuracy: float, cv_accuracy: float) -> None:
    """
    Compare train-test split vs cross-validation results
    
    Args:
        train_test_accuracy: Accuracy from train-test split
        cv_accuracy: Mean accuracy from cross-validation
    """
    print(f"\n=== Evaluation Method Comparison ===")
    print(f"Train-Test Split Accuracy: {train_test_accuracy:.4f}")
    print(f"Cross-Validation Accuracy: {cv_accuracy:.4f}")
    print(f"Difference: {abs(train_test_accuracy - cv_accuracy):.4f}")
    
    if abs(train_test_accuracy - cv_accuracy) < 0.02:
        print("✅ Results are consistent between evaluation methods")
    else:
        print("⚠️  Significant difference between evaluation methods")


def get_model_summary(model: KNeighborsClassifier, X: np.ndarray) -> Dict[str, any]:
    """
    Get summary information about the trained model
    
    Args:
        model: Trained model
        X: Feature array
    
    Returns:
        Dictionary with model summary
    """
    summary = {
        'algorithm': 'K-Nearest Neighbors',
        'n_neighbors': model.n_neighbors,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'weights': model.weights,
        'metric': model.metric
    }
    
    print(f"\n=== Model Summary ===")
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    return summary