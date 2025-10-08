# Titanic Survival Prediction ML Project

A machine learning project that predicts passenger survival on the Titanic using a K-Nearest Neighbors classifier with comprehensive feature selection and evaluation.

## 🚢 Project Overview

This project implements a complete machine learning pipeline for predicting Titanic passenger survival. The codebase follows ML engineering best practices with modular design, comprehensive testing, and reproducible results.

## 📁 Project Structure

```
project_root/
├── data/
│   ├── raw/                    # Original, immutable data
│   │   └── Titanic.csv
│   ├── processed/              # Cleaned and processed data
│   └── external/               # External data sources
├── notebooks/
│   ├── exploratory/            # Jupyter notebooks for EDA
│   │   └── 01_data_exploration.ipynb
│   ├── modeling/               # Model development notebooks
│   │   └── 02_model_training.ipynb
│   └── inference/              # Prediction and inference notebooks
│       └── 03_model_inference.ipynb
├── src/
│   ├── config/                 # Configuration files
│   │   └── settings.py
│   ├── data/                   # Data loading and preprocessing
│   │   └── preprocessing.py
│   ├── features/               # Feature engineering
│   │   └── engineering.py
│   ├── models/                 # Model definitions
│   ├── training/               # Training scripts
│   │   └── train.py
│   ├── evaluation/             # Model evaluation
│   │   └── metrics.py
│   ├── utils/                  # Utility functions
│   │   └── helpers.py
│   └── prediction/             # Prediction utilities
│       └── predictor.py
├── models/                     # Trained models and artifacts
│   ├── titanic_knn_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── significant_features.pkl
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── reports/                    # Generated reports and figures
├── train_pipeline.py           # Main training pipeline
├── requirements.txt            # Project dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml_ansh
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Training the Model

Run the complete training pipeline:
```python
python train_pipeline.py
```

This will:
- Load and preprocess the Titanic dataset
- Perform chi-square feature selection
- Train a K-Nearest Neighbors classifier
- Evaluate using both train-test split and cross-validation
- Save the trained model and preprocessing artifacts

### Making Predictions

```python
from src.prediction.predictor import TitanicPredictor

# Initialize predictor
predictor = TitanicPredictor()

# Predict for a single passenger
passenger = {
    'Pclass': 1,
    'Sex': 'female',
    'Age': 25,
    'SibSp': 0,
    'Parch': 0,
    'Embarked': 'S'
}

result = predictor.predict_single(passenger)
print(f"Prediction: {result['prediction_text']}")
print(f"Survival probability: {result['survival_probability']:.3f}")
```

## 📊 Model Performance

The K-Nearest Neighbors model achieves:
- **Cross-validation accuracy**: ~81-83%
- **Precision**: ~80-85%
- **Recall**: ~75-80%

### Features Used

The model uses statistically significant features identified through chi-square testing:
- Passenger Class (Pclass)
- Gender (Sex)
- Age
- Number of siblings/spouses (SibSp)
- Number of parents/children (Parch)
- Port of Embarkation (Embarked)

## 🔬 Methodology

### Data Preprocessing
1. **Missing Value Handling**: Rows with missing values are removed
2. **Feature Selection**: Chi-square test (α = 0.05) for statistical significance
3. **Encoding**: One-hot encoding for categorical variables
4. **Scaling**: StandardScaler for numerical features

### Model Training
- **Algorithm**: K-Nearest Neighbors (k=5)
- **Evaluation**: 5-fold cross-validation + train-test split
- **Metrics**: Accuracy, Precision, Recall

### Model Validation
- Train-test split (80/20) for initial evaluation
- 5-fold cross-validation for robust performance estimation
- Comparison between evaluation methods for consistency

## 📈 Usage Examples

### Exploratory Data Analysis
```bash
jupyter notebook notebooks/exploratory/01_data_exploration.ipynb
```

### Custom Model Training
```python
from src.training.train import train_knn_model
from src.data.preprocessing import load_raw_data

# Load data
data = load_raw_data()

# Custom training logic here...
```

### Batch Predictions
```python
import pandas as pd
from src.prediction.predictor import TitanicPredictor

predictor = TitanicPredictor()
passengers_df = pd.read_csv('new_passengers.csv')
predictions = predictor.predict(passengers_df)
```

## 🧪 Testing

Run tests:
```bash
python -m pytest tests/
```

## 📋 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- seaborn
- jupyter

See `requirements.txt` for specific versions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Kaggle for the Titanic dataset
- The scikit-learn community for excellent ML tools
- Contributors and reviewers

## 📞 Contact

For questions or suggestions, please open an issue or contact [your-email@example.com].

---

**Happy modeling! 🚢⚓**