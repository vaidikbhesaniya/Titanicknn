"""
Main training pipeline that orchestrates the entire ML workflow
"""
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

from src.data.preprocessing import (
    load_raw_data, drop_unnecessary_columns, handle_missing_values,
    encode_categorical_variables, split_features_target, scale_features,
    save_preprocessing_objects
)
from src.features.engineering import chi_square_feature_selection, filter_significant_features
from src.training.train import split_train_test, train_knn_model, save_model
from src.evaluation.metrics import (
    evaluate_train_test_split, evaluate_cross_validation, 
    compare_evaluation_methods, get_model_summary
)
from src.config.settings import MODELS_PATH


def main():
    """Main training pipeline"""
    print("🚀 Starting Titanic Survival Prediction Training Pipeline")
    print("=" * 60)
    
    # 1. Data Loading and Preprocessing
    print("\n📁 STEP 1: Data Loading and Preprocessing")
    data = load_raw_data()
    data = drop_unnecessary_columns(data)
    
    # 2. Feature Engineering and Selection
    print("\n🔧 STEP 2: Feature Engineering and Selection")
    significant_features = chi_square_feature_selection(data)
    data = filter_significant_features(data, significant_features)
    data = handle_missing_values(data)
    data = encode_categorical_variables(data)
    
    # 3. Feature and Target Separation
    print("\n📊 STEP 3: Feature and Target Separation")
    X, y = split_features_target(data)
    X_scaled, scaler = scale_features(X)
    
    # 4. Model Training
    print("\n🤖 STEP 4: Model Training")
    X_train, X_test, y_train, y_test = split_train_test(X_scaled, y)
    model = train_knn_model(X_train, y_train)
    
    # 5. Model Evaluation
    print("\n📈 STEP 5: Model Evaluation")
    y_pred = model.predict(X_test)
    
    # Train-test split evaluation
    train_test_metrics = evaluate_train_test_split(model, X_test, y_test, y_pred)
    
    # Cross-validation evaluation
    cv_metrics = evaluate_cross_validation(model, X_scaled, y)
    
    # Compare evaluation methods
    compare_evaluation_methods(train_test_metrics['accuracy'], cv_metrics['cv_accuracy_mean'])
    
    # Model summary
    model_summary = get_model_summary(model, X_scaled)
    
    # 6. Save Model and Preprocessing Objects
    print("\n💾 STEP 6: Saving Model and Preprocessing Objects")
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    save_model(model)
    save_preprocessing_objects(scaler, X.columns.tolist(), significant_features, MODELS_PATH)
    
    print("\n🎉 Training Pipeline Completed Successfully!")
    print("=" * 60)
    print(f"📋 Final Results Summary:")
    print(f"   • Train-Test Accuracy: {train_test_metrics['accuracy']:.4f}")
    print(f"   • Cross-Validation Accuracy: {cv_metrics['cv_accuracy_mean']:.4f} (±{cv_metrics['cv_accuracy_std']:.4f})")
    print(f"   • Precision: {cv_metrics['cv_precision']:.4f}")
    print(f"   • Recall: {cv_metrics['cv_recall']:.4f}")
    print(f"   • Features Used: {len(significant_features)}")
    print(f"   • Model Saved: {MODELS_PATH}")


if __name__ == "__main__":
    main()