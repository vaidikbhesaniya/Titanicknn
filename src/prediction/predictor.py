"""
Prediction utilities for making predictions on new data
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Union

from ..config.settings import (
    MODEL_FILE, SCALER_FILE, FEATURE_NAMES_FILE, 
    SIGNIFICANT_FEATURES_FILE, FEATURES_TO_DROP
)


class TitanicPredictor:
    """Class for making predictions on new Titanic data"""
    
    def __init__(self):
        """Initialize the predictor by loading saved model and preprocessing objects"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.significant_features = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all saved model artifacts"""
        try:
            # Load model
            self.model = pickle.load(open(MODEL_FILE, 'rb'))
            print(f"✅ Model loaded from {MODEL_FILE}")
            
            # Load scaler
            self.scaler = pickle.load(open(SCALER_FILE, 'rb'))
            print(f"✅ Scaler loaded from {SCALER_FILE}")
            
            # Load feature names
            self.feature_names = pickle.load(open(FEATURE_NAMES_FILE, 'rb'))
            print(f"✅ Feature names loaded from {FEATURE_NAMES_FILE}")
            
            # Load significant features
            self.significant_features = pickle.load(open(SIGNIFICANT_FEATURES_FILE, 'rb'))
            print(f"✅ Significant features loaded from {SIGNIFICANT_FEATURES_FILE}")
            
            print(f"📊 Predictor ready with {len(self.feature_names)} features")
            
        except FileNotFoundError as e:
            print(f"❌ Model artifact not found: {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading model artifacts: {e}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess new data using the same pipeline as training
        
        Args:
            data: Raw data to preprocess
            
        Returns:
            Preprocessed and scaled feature array
        """
        # Create a copy to avoid modifying original data
        data_processed = data.copy()
        
        # Drop unnecessary columns
        data_processed = data_processed.drop(FEATURES_TO_DROP, axis=1, errors='ignore')
        
        # Keep only significant features (if Survived column exists, ignore it)
        available_features = [f for f in self.significant_features if f in data_processed.columns]
        data_processed = data_processed[available_features]
        
        # Handle missing values (you might want to use more sophisticated imputation)
        data_processed = data_processed.dropna()
        
        # Encode categorical variables
        data_processed = pd.get_dummies(data_processed, drop_first=True)
        
        # Ensure all training features are present
        for feature in self.feature_names:
            if feature not in data_processed.columns:
                data_processed[feature] = 0
        
        # Reorder columns to match training data
        data_processed = data_processed[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(data_processed)
        
        print(f"✅ Data preprocessed. Shape: {X_scaled.shape}")
        return X_scaled
    
    def predict(self, data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            data: New data to predict (DataFrame or dict)
            
        Returns:
            Predictions array
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        X_scaled = self.preprocess_data(data)
        predictions = self.model.predict(X_scaled)
        
        print(f"✅ Predictions made for {len(predictions)} samples")
        return predictions
    
    def predict_proba(self, data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            data: New data to predict (DataFrame or dict)
            
        Returns:
            Prediction probabilities array
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        X_scaled = self.preprocess_data(data)
        probabilities = self.model.predict_proba(X_scaled)
        
        print(f"✅ Prediction probabilities calculated for {len(probabilities)} samples")
        return probabilities
    
    def predict_single(self, passenger_data: Dict) -> Dict:
        """
        Make prediction for a single passenger with detailed output
        
        Args:
            passenger_data: Dictionary with passenger information
            
        Returns:
            Dictionary with prediction details
        """
        prediction = self.predict(passenger_data)[0]
        probabilities = self.predict_proba(passenger_data)[0]
        
        result = {
            'prediction': int(prediction),
            'survival_probability': float(probabilities[1]),
            'death_probability': float(probabilities[0]),
            'prediction_text': 'Survived' if prediction == 1 else 'Did not survive'
        }
        
        print(f"🔮 Prediction for passenger: {result['prediction_text']}")
        print(f"   Survival probability: {result['survival_probability']:.3f}")
        
        return result


def example_prediction():
    """Example of how to use the predictor"""
    # Initialize predictor
    predictor = TitanicPredictor()
    
    # Example passenger data
    example_passenger = {
        'PassengerId': 999,
        'Pclass': 3,
        'Sex': 'male',
        'Age': 25,
        'SibSp': 0,
        'Parch': 0,
        'Embarked': 'S'
    }
    
    # Make prediction
    result = predictor.predict_single(example_passenger)
    
    return result


if __name__ == "__main__":
    # Run example prediction
    example_prediction()