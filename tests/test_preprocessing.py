"""
Test suite for data preprocessing functionality
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.preprocessing import (
    drop_unnecessary_columns,
    handle_missing_values,
    encode_categorical_variables,
    split_features_target
)


class TestPreprocessing:
    """Test cases for data preprocessing functions"""
    
    def setup_method(self):
        """Set up test data"""
        self.sample_data = pd.DataFrame({
            'Name': ['John Doe', 'Jane Smith'],
            'Pclass': [1, 3],
            'Sex': ['male', 'female'],
            'Age': [25, 30],
            'Survived': [0, 1],
            'Ticket': ['A123', 'B456'],
            'Fare': [100.0, 50.0],
            'Cabin': ['C85', None]
        })
    
    def test_drop_unnecessary_columns(self):
        """Test dropping unnecessary columns"""
        columns_to_drop = ['Name', 'Ticket', 'Fare', 'Cabin']
        result = drop_unnecessary_columns(self.sample_data, columns_to_drop)
        
        for col in columns_to_drop:
            assert col not in result.columns
        
        assert 'Pclass' in result.columns
        assert 'Sex' in result.columns
        assert 'Survived' in result.columns
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        result = handle_missing_values(self.sample_data)
        
        # Should remove rows with any missing values
        assert result.isnull().sum().sum() == 0
        assert len(result) <= len(self.sample_data)
    
    def test_encode_categorical_variables(self):
        """Test categorical variable encoding"""
        # Use data without missing values for this test
        clean_data = self.sample_data.dropna()
        result = encode_categorical_variables(clean_data)
        
        # Should have more columns due to one-hot encoding
        assert len(result.columns) >= len(clean_data.columns)
        
        # Original categorical columns should be encoded
        assert 'Sex_male' in result.columns or 'Sex_female' in result.columns
    
    def test_split_features_target(self):
        """Test feature and target splitting"""
        X, y = split_features_target(self.sample_data, 'Survived')
        
        assert 'Survived' not in X.columns
        assert len(X) == len(y)
        assert len(X.columns) == len(self.sample_data.columns) - 1
        assert y.name == 'Survived'


if __name__ == "__main__":
    pytest.main([__file__])