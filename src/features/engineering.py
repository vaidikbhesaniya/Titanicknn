"""
Feature engineering and selection utilities
"""
import pandas as pd
from scipy.stats import chi2_contingency
from typing import List, Tuple

from ..config.settings import MODEL_PARAMS


def chi_square_feature_selection(data: pd.DataFrame, target_column: str = "Survived", 
                               significance_threshold: float = MODEL_PARAMS["significance_threshold"]) -> List[str]:
    """
    Perform chi-square test for feature selection
    
    Args:
        data: DataFrame with features and target
        target_column: Name of the target column
        significance_threshold: P-value threshold for significance
    
    Returns:
        List of significant feature names
    """
    print(f"\n=== Chi-Square Feature Selection (α = {significance_threshold}) ===")
    significant_features = []
    
    for feature in data.columns:
        if feature == target_column:  # Skip target variable
            continue
        
        try:
            contingency_table = pd.crosstab(data[feature], data[target_column])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            print(f"\nFeature: {feature}")
            print("Contingency Table:")
            print(contingency_table)
            print(f"Chi2: {chi2:.3f}, P-Value: {p_value:.3f}")
            
            if p_value < significance_threshold:
                print("✅ Significant relationship → Keep this feature")
                significant_features.append(feature)
            else:
                print("❌ Not significant → Could drop this feature")
                
        except Exception as e:
            print(f"⚠️  Error processing feature {feature}: {e}")
            continue
    
    print(f"\n📊 Selected {len(significant_features)} significant features: {significant_features}")
    return significant_features


def filter_significant_features(data: pd.DataFrame, significant_features: List[str], 
                               target_column: str = "Survived") -> pd.DataFrame:
    """
    Filter dataset to keep only significant features and target
    
    Args:
        data: Original DataFrame
        significant_features: List of features to keep
        target_column: Name of the target column
    
    Returns:
        Filtered DataFrame
    """
    features_to_keep = significant_features + [target_column]
    filtered_data = data[features_to_keep]
    
    print(f"✅ Filtered data to keep {len(features_to_keep)} columns: {features_to_keep}")
    print(f"New data shape: {filtered_data.shape}")
    
    return filtered_data


def create_family_feature(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create family size feature from SibSp and Parch
    (This is an example of feature engineering - currently commented out in original code)
    
    Args:
        data: DataFrame with SibSp and Parch columns
    
    Returns:
        DataFrame with Family feature added and SibSp/Parch removed
    """
    data_copy = data.copy()
    data_copy['Family'] = data_copy['SibSp'] + data_copy['Parch']
    data_copy = data_copy.drop(['SibSp', 'Parch'], axis=1)
    
    print("✅ Created Family feature from SibSp + Parch")
    return data_copy