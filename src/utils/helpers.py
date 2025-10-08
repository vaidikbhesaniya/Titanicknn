"""
Utility functions for the Titanic ML project
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional


def display_data_info(data: pd.DataFrame, title: str = "Dataset Information") -> None:
    """
    Display comprehensive information about the dataset
    
    Args:
        data: DataFrame to analyze
        title: Title for the information display
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    print(f"📊 Shape: {data.shape}")
    print(f"📈 Memory usage: {data.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    print(f"\n📋 Column Information:")
    for col in data.columns:
        dtype = data[col].dtype
        null_count = data[col].isnull().sum()
        null_pct = (null_count / len(data)) * 100
        unique_count = data[col].nunique()
        
        print(f"   {col:<15} | {str(dtype):<10} | Nulls: {null_count:>3} ({null_pct:>5.1f}%) | Unique: {unique_count:>3}")
    
    print(f"\n📈 Statistical Summary:")
    if len(data.select_dtypes(include=[np.number]).columns) > 0:
        print(data.describe())


def create_data_quality_report(data: pd.DataFrame) -> Dict:
    """
    Create a comprehensive data quality report
    
    Args:
        data: DataFrame to analyze
        
    Returns:
        Dictionary with data quality metrics
    """
    report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().to_dict(),
        'missing_percentage': ((data.isnull().sum() / len(data)) * 100).to_dict(),
        'duplicate_rows': data.duplicated().sum(),
        'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(data.select_dtypes(include=['object']).columns),
        'data_types': data.dtypes.to_dict()
    }
    
    return report


def save_results_summary(metrics: Dict, file_path: Path) -> None:
    """
    Save model results summary to a text file
    
    Args:
        metrics: Dictionary containing model metrics
        file_path: Path to save the summary
    """
    with open(file_path, 'w') as f:
        f.write("Titanic Survival Prediction - Model Results Summary\n")
        f.write("="*50 + "\n\n")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
    
    print(f"✅ Results summary saved to: {file_path}")


def format_feature_importance(feature_names: List[str], model_coef: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Format feature importance for display (placeholder for KNN which doesn't have feature importance)
    
    Args:
        feature_names: List of feature names
        model_coef: Model coefficients (not applicable for KNN)
        
    Returns:
        DataFrame with feature information
    """
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Type': ['Numerical' if 'float' in str(type(f)) else 'Categorical' for f in feature_names],
        'Note': ['Used in KNN model'] * len(feature_names)
    })
    
    return feature_df


def validate_input_data(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that input data has required columns
    
    Args:
        data: Input DataFrame
        required_columns: List of required column names
        
    Returns:
        True if valid, False otherwise
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return False
    
    print("✅ Input data validation passed")
    return True


def create_prediction_report(predictions: np.ndarray, probabilities: Optional[np.ndarray] = None) -> Dict:
    """
    Create a summary report for predictions
    
    Args:
        predictions: Array of predictions
        probabilities: Array of prediction probabilities (optional)
        
    Returns:
        Dictionary with prediction summary
    """
    report = {
        'total_predictions': len(predictions),
        'survived_count': int(np.sum(predictions == 1)),
        'died_count': int(np.sum(predictions == 0)),
        'survival_rate': float(np.mean(predictions))
    }
    
    if probabilities is not None:
        report['avg_survival_probability'] = float(np.mean(probabilities[:, 1]))
        report['confidence_above_80'] = int(np.sum(np.max(probabilities, axis=1) > 0.8))
    
    return report


def log_experiment(experiment_name: str, metrics: Dict, log_dir: Path) -> None:
    """
    Log experiment results for tracking
    
    Args:
        experiment_name: Name of the experiment
        metrics: Dictionary of metrics to log
        log_dir: Directory to save logs
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{experiment_name}_log.txt"
    
    with open(log_file, 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n")
        f.write("-" * 40 + "\n")
        
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"📝 Experiment logged to: {log_file}")