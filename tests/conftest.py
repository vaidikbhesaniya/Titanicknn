# Test configuration and shared fixtures
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_titanic_data():
    """Create sample Titanic data for testing"""
    return pd.DataFrame({
        'PassengerId': [1, 2, 3, 4],
        'Pclass': [1, 3, 2, 1],
        'Name': ['John Doe', 'Jane Smith', 'Bob Wilson', 'Alice Brown'],
        'Sex': ['male', 'female', 'male', 'female'],
        'Age': [25, 30, 35, 28],
        'SibSp': [0, 1, 0, 1],
        'Parch': [0, 0, 2, 1],
        'Ticket': ['A123', 'B456', 'C789', 'D012'],
        'Fare': [100.0, 50.0, 75.0, 120.0],
        'Cabin': ['C85', None, 'B42', 'A12'],
        'Embarked': ['S', 'C', 'Q', 'S'],
        'Survived': [0, 1, 1, 0]
    })


@pytest.fixture
def project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(project_root):
    """Get test data directory"""
    return project_root / "tests" / "data"