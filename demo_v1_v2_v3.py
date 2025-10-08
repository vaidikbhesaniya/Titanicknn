"""
Demo script showing V1, V2, V3 components with actual console output
This script uses the correct file paths after restructuring
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.stats import chi2_contingency

print("🚢 Titanic ML Project - V1, V2, V3 Component Demonstration")
print("=" * 60)

# Load dataset (updated path)
try:
    data = pd.read_csv("data/raw/Titanic.csv")
    print(f"✅ Dataset loaded successfully. Shape: {data.shape}")
except FileNotFoundError:
    print("❌ Titanic.csv not found in data/raw/. Please ensure the file is in the correct location.")
    exit(1)

# Drop unnecessary columns
data = data.drop(['Name', 'Ticket', 'Fare', 'Cabin'], axis=1)
print(f"✅ Unnecessary columns dropped. New shape: {data.shape}")

print("\n" + "="*60)
print("V2: CHI-SQUARE FEATURE SELECTION")
print("="*60)

significant_features = []

for feature in data.columns:
    if feature == "Survived":  # Skip target variable
        continue
    
    contingency_table = pd.crosstab(data[feature], data["Survived"])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\nFeature: {feature}")
    print("Contingency Table:")
    print(contingency_table)
    print(f"Chi2: {chi2:.3f}, P-Value: {p_value:.3f}")
    
    if p_value < 0.05:
        print("✅ Significant relationship → Keep this feature")
        significant_features.append(feature)
    else:
        print("❌ Not significant → Could drop this feature")

print(f"\n📊 SELECTED FEATURES: {significant_features}")
print(f"Total significant features: {len(significant_features)}")

# Keep only significant features + target
data = data[significant_features + ["Survived"]]

# Drop rows with missing values
data = data.dropna()
print(f"✅ Missing values handled. Final data shape: {data.shape}")

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)
print(f"✅ Categorical variables encoded. Final shape: {data.shape}")

# Split features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✅ Features standardized. Feature matrix shape: {X_scaled.shape}")

print("\n" + "="*60)
print("V1: TRAIN-TEST SPLIT EVALUATION")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Test split ratio: 20%")

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== V1: Train-Test Split Metrics ===")
train_test_accuracy = accuracy_score(y_test, y_pred)
train_test_precision = precision_score(y_test, y_pred)
train_test_recall = recall_score(y_test, y_pred)

print(f"Accuracy:  {train_test_accuracy:.4f} ({train_test_accuracy*100:.2f}%)")
print(f"Precision: {train_test_precision:.4f} ({train_test_precision*100:.2f}%)")
print(f"Recall:    {train_test_recall:.4f} ({train_test_recall*100:.2f}%)")

# Save Model and Preprocessing Objects
print("\n=== Saving Model and Preprocessing Objects ===")

# Ensure models directory exists
import os
os.makedirs('./models', exist_ok=True)

# Save the trained model
pickle.dump(model, open('./models/titanic_knn_model.pkl', 'wb'))
print("✅ Model saved as 'titanic_knn_model.pkl'")

# Save the scaler for consistent preprocessing
pickle.dump(scaler, open('./models/scaler.pkl', 'wb'))
print("✅ Scaler saved as 'scaler.pkl'")

# Save feature names for future use
feature_names = X.columns.tolist()
pickle.dump(feature_names, open('./models/feature_names.pkl', 'wb'))
print("✅ Feature names saved as 'feature_names.pkl'")

# Save significant features list
pickle.dump(significant_features, open('./models/significant_features.pkl', 'wb'))
print("✅ Significant features saved as 'significant_features.pkl'")

print("\n🎉 All pickle files generated successfully!")

print("\n" + "="*60)
print("V3: TRAIN-TEST SPLIT vs K-FOLD CROSS-VALIDATION")
print("="*60)

print(f"Train-Test Split Accuracy: {train_test_accuracy:.4f} ({train_test_accuracy*100:.2f}%)")

# K-Fold Cross Validation (Accuracy)
print("\nPerforming 5-Fold Cross-Validation...")
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
print(f"\nCross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} ({np.mean(cv_scores)*100:.2f}%)")
print(f"CV Standard Deviation: {np.std(cv_scores):.4f} ({np.std(cv_scores)*100:.2f}%)")

# Cross-validation with precision & recall
y_cv_pred = cross_val_predict(model, X_scaled, y, cv=5)

print("\n=== V3: Cross-Validation Metrics ===")
cv_accuracy = accuracy_score(y, y_cv_pred)
cv_precision = precision_score(y, y_cv_pred)
cv_recall = recall_score(y, y_cv_pred)

print(f"Accuracy:  {cv_accuracy:.4f} ({cv_accuracy*100:.2f}%)")
print(f"Precision: {cv_precision:.4f} ({cv_precision*100:.2f}%)")
print(f"Recall:    {cv_recall:.4f} ({cv_recall*100:.2f}%)")

print("\n" + "="*60)
print("COMPARISON ANALYSIS")
print("="*60)

print("Metric Comparison:")
print(f"{'Metric':<12} | {'Train-Test':<12} | {'Cross-Val':<12} | {'Difference':<12}")
print("-" * 55)
print(f"{'Accuracy':<12} | {train_test_accuracy:<12.4f} | {cv_accuracy:<12.4f} | {abs(train_test_accuracy-cv_accuracy):<12.4f}")
print(f"{'Precision':<12} | {train_test_precision:<12.4f} | {cv_precision:<12.4f} | {abs(train_test_precision-cv_precision):<12.4f}")
print(f"{'Recall':<12} | {train_test_recall:<12.4f} | {cv_recall:<12.4f} | {abs(train_test_recall-cv_recall):<12.4f}")

# Model stability assessment
accuracy_diff = abs(train_test_accuracy - cv_accuracy)
if accuracy_diff < 0.02:
    stability = "✅ STABLE - Results are consistent"
elif accuracy_diff < 0.05:
    stability = "⚠️  MODERATE - Some variation present"
else:
    stability = "❌ UNSTABLE - Significant variation detected"

print(f"\nModel Stability Assessment: {stability}")
print(f"Accuracy difference: {accuracy_diff:.4f} ({accuracy_diff*100:.2f}%)")

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"📊 Model Performance Summary:")
print(f"   • Algorithm: K-Nearest Neighbors (k=5)")
print(f"   • Features Used: {len(significant_features)} statistically significant features")
print(f"   • Training Samples: {len(X_scaled)}")
print(f"   • Cross-Validated Accuracy: {cv_accuracy:.4f} (±{np.std(cv_scores):.4f})")
print(f"   • Model Status: {stability.split(' - ')[1] if ' - ' in stability else stability}")

print(f"\n🎯 Key Insights:")
print(f"   • Chi-square test identified {len(significant_features)} significant predictors")
print(f"   • Model shows {'consistent' if accuracy_diff < 0.02 else 'variable'} performance across evaluation methods")
print(f"   • Cross-validation provides more robust performance estimate")

print(f"\n🚀 Model is ready for deployment!")
print("=" * 60)